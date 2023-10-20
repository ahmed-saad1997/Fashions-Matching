import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import ImageTk, Image
from FeatureExtraction import *
from Segmentation import *
from VectorDatabase import *
import torch

device='cuda' if torch.cuda.is_available()  else 'cpu'

Seg_model=SegmentationModel(device=device)
feat_model=FeatureExtractionModel(device=device)
vec_model=VectorDatabaseModel(feat_model,Seg_model,'vector_database')

def get_similar_images(img_path):
    masks, classes,masked_img= Seg_model.predict(img_path,plot=True)
    if len(masked_img)==0:
        return []
    images_res=[]
    for m, cls in zip(masks, classes):
        images = vec_model.get_simillers(m,cls,k=2)
        images_res.extend([im[0] for im in images])
    return images_res,masked_img

def exit_app():
    window.destroy()

def process_image(file_path):
    for i in range(len(similar_image_labels)):
        similar_image_labels[i].destroy()
    similar_image_labels.clear()
    input_image = Image.open(file_path)
    input_image = input_image.resize((400, 400))
    input_image = ImageTk.PhotoImage(input_image)
    input_image_label.configure(image=input_image)
    input_image_label.image = input_image
    similar_images,masked_img = get_similar_images(file_path)
    if len(similar_images)==0:
        return
    segmentation_mask =cv2.resize(masked_img,(400,400))
    segmentation_mask=Image.fromarray(cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2RGB))
    segmentation_mask = ImageTk.PhotoImage(segmentation_mask)
    segmentation_mask_label.configure(image=segmentation_mask)
    segmentation_mask_label.image = segmentation_mask
    for _ in range(len(similar_images)):
        similar_image_label = tk.Label(similar_images_frame, background="#FFF")
        similar_image_label.pack(side=tk.LEFT, padx=10)
        similar_image_label.configure(relief="solid", bd=1, padx=8, pady=8)
        similar_image_labels.append(similar_image_label)
    for i, image_path in enumerate(similar_images):
        similar_image = Image.open(image_path)
        similar_image = similar_image.resize((150, 150))
        similar_image = ImageTk.PhotoImage(similar_image)
        similar_image_labels[i].configure(image=similar_image)
        similar_image_labels[i].image = similar_image


def select_image():
    # Open a file dialog to allow the user to select an image file
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if file_path:
        process_image(file_path)


# Create the main window
window = tk.Tk()
window.title("Fashion Piece Segmentation")
window.configure(background="#FFF")
window.attributes('-fullscreen',True)


# Create labels to display the images
input_frame = tk.Frame(window, background="#FFF")
input_frame.pack(pady=20)

input_image_label = tk.Label(input_frame, background="#FFF")
input_image_label.pack(pady=10, padx=20, side=tk.LEFT)
input_image_placeholder= Image.open('placeholder.png')
input_image_placeholder=input_image_placeholder.resize((400, 400))
input_image_placeholder = ImageTk.PhotoImage(input_image_placeholder)
input_image_label.configure(image=input_image_placeholder)

segmentation_mask_label = tk.Label(input_frame, background="#FFF")
segmentation_mask_label.pack(pady=10, padx=20, side=tk.LEFT)
segmentation_mask_label.configure(image=input_image_placeholder)


similar_images_frame = tk.Frame(window, background="#FFF")
similar_images_frame.pack(pady=20)

label = tk.Label(similar_images_frame, text="Similar Items from Stock",font=("Helvetica", 14, "bold"))
label.pack(padx=10,pady=5,side=tk.TOP)

similar_image_placeholder = tk.Label(similar_images_frame, background="#FFF")
similar_image_placeholder.pack(side=tk.LEFT, padx=10)
similar_image = Image.open('placeholder.png')
similar_image = similar_image.resize((150, 150))
similar_image = ImageTk.PhotoImage(similar_image)
similar_image_placeholder.configure(image=similar_image)
similar_image_placeholder1 = tk.Label(similar_images_frame, background="#FFF")
similar_image_placeholder1.pack(side=tk.LEFT, padx=10)
similar_image_placeholder1.configure(image=similar_image)

similar_image_labels = [similar_image_placeholder,similar_image_placeholder1]
# Create a button to select an image

buttons_frame = tk.Frame(window, background="#FFF")
buttons_frame.pack(pady=5)

select_button = tk.Button(
    buttons_frame,
    text="Select Image",
    command=select_image,
    background="#48A999",
    foreground="#FFF",
    font=("Helvetica", 14, "bold"),
    relief="flat",
    padx=10,
    pady=10
)
select_button.pack(side=tk.LEFT)

exitbutton = tk.Button(
    buttons_frame,
    text="Exit",
    command=exit_app,
    background="#8b0000",
    foreground="#FFF",
    font=("Helvetica", 14, "bold"),
    relief="flat",
    padx=10,
    pady=10
)
exitbutton.pack(padx=20,side=tk.RIGHT)

# Start the main event loop
window.mainloop()
