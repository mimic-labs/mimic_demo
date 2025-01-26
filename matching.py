import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from scipy.ndimage import binary_closing, binary_opening
import torchvision.transforms as transforms
from typing import Tuple
from PIL import ImageDraw
import gc
from tqdm import tqdm

########################## SAM2/DinoV2 Initializations ##########################
def initialize_sam2(device):
    sam2_checkpoint = "./third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

    mask_predictor = SAM2AutomaticMaskGenerator(sam2)
    prompted_predictor = SAM2ImagePredictor(sam2)
    return sam2, mask_predictor, prompted_predictor

def initialize_dinov2(device, REPO_NAME = "facebookresearch/dinov2", MODEL_NAME = "dinov2_vitb14"):
    model = torch.hub.load(repo_or_dir=REPO_NAME, model=MODEL_NAME)
    model.to(device)
    model.eval()
    return model
################################################################################


########################## SAM2 Helper Functions ##########################
def generate_masks(mask_generator, prompted_predictor, image_path, point_prompt=None, box_prompt=None):
    """
    mask_generator: output of initialize_sam2 for all mask generation
    prompted_predictor: output of initialize_sam2 for box and points prompted generation
    image_path: path of image
    point_prompt: np.array(b,n,2), where n is # of points
    box_prompt: np.array(b,4), where n is # of boxes
    More info on input prompts:
        https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/sam2_image_predictor.py#L353
    """
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))
    if point_prompt is None and box_prompt is None:
        masks = mask_generator.generate(image)
        valid_indices = deduplicate_masks(masks)
        torch.cuda.empty_cache()
        return [masks[i] for i in range(len(masks)) if i in valid_indices]
    else:
        prompted_predictor.set_image(image)
        masks, scores, logits = prompted_predictor.predict(
            point_coords=point_prompt,
            box=box_prompt,
            multimask_output=True,
            point_labels=torch.ones(point_prompt.shape[:2]),
        )
        torch.cuda.empty_cache()

        return masks, scores, logits

def deduplicate_masks(masks, threshold=0.7):
    valid = set(list(range(len(masks))))
    for i in range(len(masks)):
        for j in range(i+1, len(masks)):
            mask1, mask2 = masks[i]['segmentation'], masks[j]['segmentation']
            iou = (mask1*mask2).sum()/(mask1+mask2).sum()
            if iou >= threshold:
                if j in valid:
                    valid.remove(j)
                elif i in valid:
                    valid.remove(i)
    return valid
    
def create_mask_images(original_image, masks):
    """
    original_image: Image
    masks: np.array of each mask in original_image

    returns: PIL images of each mask
    """
    original_image_array = np.array(original_image)
    all_cropped_imgs = []
    B,N,_,_ = masks.shape
    for i in range(B):
        per_batch = []
        for j in range(N):
            mask = masks[i, j, :, :]
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            np_mask = mask.unsqueeze(dim=2).numpy()
            masked_img = (original_image_array * np_mask).astype(np.uint8)  # Convert to uint8
            ys, xs = np.where(mask)
            if ys.size == 0 or xs.size == 0:
                continue
            bbox = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
            cropped_image_array = masked_img[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1, :]
            cropped_image_pil = Image.fromarray(cropped_image_array)
            per_batch.append(cropped_image_pil)
        all_cropped_imgs.append(per_batch)

    return all_cropped_imgs
################################################################################


########################## DinoV2 Helper Functions ##########################
def make_transform(smaller_edge_size: int) -> transforms.Compose:
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    interpolation_mode = transforms.InterpolationMode.BICUBIC

    return transforms.Compose([
        transforms.Resize(size=smaller_edge_size, interpolation=interpolation_mode, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

def prepare_image(image: Image,
                  smaller_edge_size: float,
                  patch_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    transform = make_transform(int(smaller_edge_size))
    image_tensor = transform(image)

    # Crop image to dimensions that are a multiple of the patch size
    height, width = image_tensor.shape[1:] # C x H x W
    cropped_width, cropped_height = width - width % patch_size, height - height % patch_size
    image_tensor = image_tensor[:, :cropped_height, :cropped_width]

    grid_size = (cropped_height // patch_size, cropped_width // patch_size) # h x w (TODO: check)
    scale_width = image.width / cropped_width
    scale_height = image.height / cropped_height
    return image_tensor, grid_size, (scale_width, scale_height)

def get_obj_embeddings(model, cropped_obj):
    """
    model - dinov2 model
    cropped_obj: 1 Image
    """

    image_tensor, grid_size, scales = prepare_image(cropped_obj, 448, 14)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    with torch.no_grad():
        tokens, cls = model.get_intermediate_layers(image_tensor.unsqueeze(0).to(device), return_class_token=True)[0]
    return tokens.detach().cpu().squeeze(), cls.detach().cpu().squeeze(), (grid_size, scales), image_tensor.detach().cpu()
################################################################################

########################## General Helper Functions ##########################
def calculate_simmatrix(a, b, eps=1e-8):
    """
    a: BxNxD
    b: MxD
    
    out: NxM, each row contains how similar N_i is to each M
    """
    a_n, b_n = a.norm(dim=-1,keepdim=True), b.norm(dim=-1,keepdim=True)
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = a_norm @ b_norm.T
    return sim_mt
    
def match_ref_and_query(query_embeddings, reference_embeddings):
    """
    query_embedding: NxD
    returns closest reference_embedding match to each query_embedding

    """
    similarities = calculate_simmatrix(query_embeddings, reference_embeddings)
    matched_ref_masks, matched_ref_masks_idx = torch.max(similarities, dim=-1)
    # self.matched_query_masks = self.query_masks[matched_query_masks_idx, :, :]
    # self.matched_query_patch_embeddings = self.query_patch_embeddings[matched_query_masks_idx, :, :]
    return matched_ref_masks_idx, matched_ref_masks
################################################################################

############################################# Helper Funcs: Matching Keypoints on ImageResults #################################################
# Given a pair of ImageResults
def source_position_to_idx(row, col, scales):
    grid_size, (scale_width, scale_height) = scales
    idx = ((row / scale_height) // (14)) * grid_size[1] + ((col / scale_width) // (14))
    return int(idx)

def idx_to_source_position(idx, scales):
    grid_size, (scale_width, scale_height) = scales
    row = (idx // grid_size[1])*14*scale_height + 14 / 2
    col = (idx % grid_size[1])*14*scale_width + 14 / 2
    return int(row), int(col)

def closest_embedding(ref_embedding, query_embeddings, query_mask, method="dist_norm"):
    """
    ref_embedding: 1xD
    query_embeddings: NxD
    query_mask: (N,)
    """
    if method == "dist_norm":
        distances = torch.norm(query_embeddings - ref_embedding, dim=1)
    else: 
        distances = 1 - calculate_simmatrix(ref_embedding, query_embeddings).flatten()
    dist_copy = distances.clone()
    distances[~query_mask] = float('inf')
    return torch.argmin(distances).item(), dist_copy

def generate_heatmap(distances, mask, grid_size, image_size):
    distances = distances.reshape(grid_size)
    mask = mask.reshape(grid_size)
    heatmap_np = distances.numpy()
    heatmap_np *= -1 #convert distance matrix -> similarity matrix
    heatmap_np = (heatmap_np - np.min(heatmap_np)) / (np.max(heatmap_np) - np.min(heatmap_np))
    heatmap_np[~mask] = 0

    cmap = plt.get_cmap('jet')
    heatmap = cmap(heatmap_np)
    heatmap_rgb = (heatmap[:, :, :3] * 255).astype(np.uint8)
    resized_heatmap = cv2.resize(heatmap_rgb, (image_size[0], image_size[1]))
    
    return resized_heatmap, heatmap_np

def make_foreground_mask(image_tensor):
    def zero_pixel(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        val = mean[0] / std[0] + mean[1] / std[1] + mean[2]/std[2]
        return -1 * val
    mask = torch.sum(image_tensor, dim=0)
    threshold = zero_pixel()
    mask = (torch.abs(mask - threshold) > 0.001).int()
    new_size = (mask.size(0) // 14, mask.size(1) // 14)
    resized_mask = torch.empty(new_size, dtype=torch.bool)
    for i in range(new_size[0]):
        for j in range(new_size[1]):
            ones = torch.sum(mask[i*14:(i+1)*14, j*14:(j+1)*14])
            if ones <= (14 * 14 * 0.8):
                resized_mask[i, j] = False
            else:
                resized_mask[i, j] = True
    
    mask = resized_mask.flatten()
    return mask.flatten()

def get_corresponding_contacts(ref, query, contact_point):
    """
    ref: reference ImageResults
    query: query ImageResults
    contact_point: list([B,#ofptsperbatch,2]); [x,y] in full reference image space

    ref and query should now contain 3 new attributes each:
    - image_with_contact: image with contact_point draw (corresponding point if it is the query image)
    - cropped_image_space_coords: contact point [x,y] wrt cropped image
    - orig_image_space_coords: contact point [x,y] wrt full/original image
    """

    ref.image_with_contact = ref.orig_image.copy()
    query.image_with_contact = query.orig_image.copy()
    ref.cropped_image_space_coords = []
    ref.orig_image_space_coords = []
    query.cropped_image_space_coords = []
    query.orig_image_space_coords = []
    query.heatmap = []
    query.heatmap_np = []
    
    for best in range(len(contact_point)):
        refcropped_image_space_coords_pb = []
        reforig_image_space_coords_pb = []
        querycropped_image_space_coords_pb = []
        queryorig_image_space_coords = []
        queryheatmap_pb = []
        queryheatmap_np_pb = []
        for pt in contact_point[best]:
            draw = ImageDraw.Draw(ref.image_with_contact)
            ys, xs = np.where(ref.mask[best])
            box_x, box_y = np.min(xs), np.min(ys)
            col, row = pt
            draw.ellipse([col-10, row-10, col+10, row+10], fill=(255, 0, 0))
            
            contact_pt = [row-box_y,col-box_x] # [row, col] = [y, x]
            refcropped_image_space_coords_pb.append(contact_pt[::-1])
            reforig_image_space_coords_pb.append(pt)
            
            query_mask = make_foreground_mask(query.image_tensor[best])
            idx = source_position_to_idx(contact_pt[0], contact_pt[1], ref.scales[best])
            matched_idx, distances = closest_embedding(ref.cropped_image_tokens[best][[idx], :], query.cropped_image_tokens[best], query_mask, method="")
            heatmap, heatmap_np = generate_heatmap(distances, query_mask, query.scales[best][0], query.cropped_image[best].size)
            row, col = idx_to_source_position(matched_idx, query.scales[best])
            queryheatmap_pb.append(heatmap)
            queryheatmap_np_pb.append(heatmap_np)
            
            draw = ImageDraw.Draw(query.image_with_contact)
            ys, xs = np.where(query.mask[best])
            box_x, box_y = np.min(xs), np.min(ys)
            draw.ellipse([box_x+col-10, box_y+row-10, box_x+col+10, box_y+row+10], fill=(255, 0, 0))
            querycropped_image_space_coords_pb.append([col,row])
            queryorig_image_space_coords.append([box_x+col, box_y+row])
        ref.cropped_image_space_coords.append(refcropped_image_space_coords_pb)
        ref.orig_image_space_coords.append(reforig_image_space_coords_pb)
        query.cropped_image_space_coords.append(querycropped_image_space_coords_pb)
        query.orig_image_space_coords.append(queryorig_image_space_coords)
        query.heatmap.append(queryheatmap_pb)
        query.heatmap_np.append(queryheatmap_np_pb)
    
###############################################################################################################################################


class ImageResults:
    def __init__(self, orig_image, cropped_image, image_tensor, mask, cropped_image_tokens, cropped_image_cls, scales, **kwargs):
        
        self.orig_image = orig_image # original PIL image
        self.cropped_image = cropped_image # cropped mask PIL image (this could be either selected mask or matched mask image)
        self.image_tensor = image_tensor # post transformation image tensor
        self.mask = mask # binary mask of mask in image
        self.cropped_image_cls = cropped_image_cls # 1x768, embedding_dim=768
        self.cropped_image_tokens = cropped_image_tokens # Nx768, N = # of 14x14 blocks
        self.scales = scales

        # Optional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def attributes(self):
        return self.__dict__

    def __getitem__(self, key):
        # Get attribute value using dictionary-like indexing
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise KeyError(f"'{key}' not found in Results attributes")
        
    def __setitem__(self, key, value):
        # Set attribute value using dictionary-like indexing
        self.__dict__[key] = value
      

def get_mask1_bestmask2(models, image_path1, image_path2, pos_points):
    """
    models = (dinov2, mask_predictor, prompted_predictor)

    pos_points: torch.tensor(B,N,2) of points

    Returns mask in image1 and best corresponding mask in image2
    """
    dinov2, mask_predictor, prompted_predictor = models
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    # Mask Generation
    masks1, scores, logits = generate_masks(mask_predictor, prompted_predictor, image_path1, point_prompt=pos_points) # point prompt for image1
    best_mask1 = masks1 if len(masks1.shape)==4 else masks1[None,:,:,:] #after: top 3 masks before: top 1 mask[[np.argmax(scores)]] 
    masks2 = generate_masks(mask_predictor, prompted_predictor, image_path2) # get all masks in image2
    masks2 = np.array([mask['segmentation'] for mask in masks2])
    torch.cuda.empty_cache()
    gc.collect()
    print("done with mask generation")
    
    # PIL.Image Generation from masks
    cropped_images1 = create_mask_images(image1, best_mask1) #[B, 3]
    cropped_images2 = create_mask_images(image2, masks2[None,:,:,:])[0]

    # Calculate DinoV2 embeddings for all masks and get best match in image2
    # query_tokens, query_cls, query_scales = get_obj_embeddings(dinov2, cropped_images1[0])
    # query_cls = queries_cls.unsqueeze(0) #get embedding for touched mask in image1; 1,768
    query_tokens = []
    query_cls = []
    query_scales = [] 
    query_image_tensor = []

    for batch in cropped_images1:
        query_embs = [get_obj_embeddings(dinov2, c) for c in batch]
        qt, qc, qs, qit = zip(*query_embs)
        query_tokens.append(list(qt))
        query_cls.append(torch.stack(list(qc)))
        query_scales.append(list(qs))
        query_image_tensor.append(list(qit))
        torch.cuda.empty_cache()
        gc.collect()

    refs_embs = []
    for i, c in enumerate(tqdm(cropped_images2)):
        refs_embs.append(get_obj_embeddings(dinov2, c))
        torch.cuda.empty_cache()
        gc.collect()
    refs_tokens, refs_cls, refs_scales, refs_image_tensor = zip(*refs_embs)
    refs_tokens = list(refs_tokens)
    refs_cls = torch.stack(list(refs_cls))
    refs_scales = list(refs_scales)
    refs_image_tensor = list(refs_image_tensor)
    
    idxs, maxs = match_ref_and_query(torch.stack(query_cls), refs_cls) #[B, 3] for each batch, for each query, match to 1 reference
    best = torch.argmax(maxs, dim=-1) # best score out of top score for each of the 3 masks
    print(f"Out of the top 3 masks, the best ones were {best.tolist()} (index per batch)")
    idxs = idxs[np.arange(len(idxs)), best].numpy() # shape: [B,]
    
    resultsIm1 = ImageResults(image1, 
                              [cropped_images1[i][best[i]] for i in range(len(best))], 
                              [query_image_tensor[i][best[i]] for i in range(len(best))], 
                              [best_mask1[i][best[i]] for i in range(len(best))], 
                              [query_tokens[i][best[i]] for i in range(len(best))], 
                              [query_cls[i][best[i]] for i in range(len(best))], 
                              [query_scales[i][best[i]] for i in range(len(best))], 
                              pos_points=pos_points)
    
    resultsIm2 = ImageResults(image2, 
                              [cropped_images2[idx] for idx in idxs],
                              [refs_image_tensor[idx] for idx in idxs],
                              masks2[idxs], 
                              [refs_tokens[idx] for idx in idxs], 
                              refs_cls[idxs],
                              [refs_scales[idx] for idx in idxs])
    return resultsIm1, resultsIm2