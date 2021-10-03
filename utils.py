from torchvision.utils import draw_bounding_boxes
from torch import Tensor, max, from_numpy, as_tensor, int64
from torchvision.transforms import ToTensor, ToPILImage
from numpy import ndarray, array
from PIL.Image import Image
from torchvision.transforms import Compose, RandomPosterize, \
RandomAdjustSharpness, RandomAutocontrast, GaussianBlur
import matplotlib.pyplot as plt
from numpy.random import uniform
from torch.nn import Module
from torchvision.transforms.functional import pil_to_tensor, convert_image_dtype
                                    

def obj_detect_visualization(image, 
                  coord,
                  wh_to_diag_coord = True,
                  input_labels = None,
                  input_colors = None,
                  input_fill = False,
                  input_width = 1,
                  input_font = None,
                  input_font_size = 10):
    """ 
    
    Visualization function for object detection:
        
        -image(torch.Tensor/np.ndarray/PIL.Image.Image/list): PIL image, numpy tensor, python built-in list or torch tensor.
        
        -coord(torch.Tensor/np.ndarray/list): coordinate of bounding boxes in form of top left coordinate 
        and lower right coordinate, input using 2D torch tensor, 2D numpy 
        array or Python 3 array.
        
        -wh_to_diag_coord (bool, Default = True): convert coord form top left 
        coordinate with wide and height of bounding box to top left coordinate 
        and lower right coordinate.
        
        -input_labels (List[str]): List containing the labels of bounding boxes.
        
        -input_colors (List[Union[str, Tuple[int, int, int]]]): List containing the colors of bounding boxes. The colors can
            be represented as `str` or `Tuple[int, int, int]`.
            
        -input_fill (bool): If `True` fills the bounding box with specified color.
        
        -input_width (int): Width of bounding box.
        
        -input_font (str): A filename containing a TrueType font. If the file is not found in this filename, the loader may
        also search in other directories, such as the `fonts/` directory on Windows or `/Library/Fonts/`,
        `/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
        
        -input_font_size (int): The requested font size in points.

    Returns:
        PIL image with ploted box.
 
    """
    
    def to_diag_coord(tensors):
        for tensor in tensors:
            tensor[2] = tensor[2].add(tensor[0])
            tensor[3] = tensor[3].add(tensor[1])
        return tensors
    
    #Convert data type to uint8 torch.Tensor
    
    # For image 
    if isinstance(image, list):
        true_input = (Tensor(image)*255).byte()
    elif isinstance(image, Image):
        true_input = (ToTensor()(image)*255).byte()
    elif isinstance(image, Tensor):
        if (max(image)>1): true_input = image.byte()
        else: true_input = (image*255).byte()
    elif isinstance(image, ndarray):
        temp = from_numpy(image)
        if (max(temp)>1): true_input = temp.byte()
        else: true_input = (temp*255).byte()
    
    #For coordinate   
    if isinstance(coord, list):
        coordinate = Tensor(coord)
    elif isinstance(coord, ndarray):
        coordinate = from_numpy(coord)
    elif isinstance(image, Tensor):
        coordinate = coord
        
    #Coordinate transformation
    if wh_to_diag_coord:
        true_coord = to_diag_coord(coordinate)
    
    #Apply bounding box
    result = draw_bounding_boxes(true_input, 
                                 true_coord,
                                 labels = input_labels,
                                 colors = input_colors,
                                 fill = input_fill,
                                 width = input_width,
                                 font = input_font,
                                 font_size = input_font_size)
    
    return ToPILImage()(result)

class RandomGausianBlur(Module):

    def __init__(self, 
                 input_kernel_size, 
                 input_sigma, 
                 random_threshold=0.5):
        super().__init__()
        self.g_blur = GaussianBlur(kernel_size=input_kernel_size, 
                                   sigma=input_sigma)
        self.random_threshold = random_threshold

    def forward(self, img):
        if uniform() > self.random_threshold:
            img = self.g_blur(img)
        
        return img

def get_image_aug(): 
    """
    Compose a sequence of augmentation for image in 
    image segmentation (after being rescaled)
    """    
    transform = Compose([
        RandomAdjustSharpness(sharpness_factor=2),
        RandomAutocontrast(),
        RandomGausianBlur(input_kernel_size=(5, 9), 
                          input_sigma=(0.1, 5))
    ])
    
    return transform

def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(ToPILImage()(display_list[i]))
        plt.axis('off')
        
    plt.show()
    
class PILtoTensor(object):
    """Turn PIL image into tensor with speficied dtype.

    Args:
       numpy_dtype (numpy dtype): data type in used in numpy, check
       https://numpy.org/doc/stable/user/basics.types.html for more info.
    """

    def __init__(self,
                numpy_dtype=None):
        self.dtype = numpy_dtype
        
    def __call__(self, image):
        assert isinstance(image, Image)
        image = array(image)
        if self.dtype is not None: image = image.astype(self.dtype)
        return from_numpy(image).unsqueeze(0)

    
