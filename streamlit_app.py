import streamlit as st
import requests
from PIL import Image
import glob
from io import BytesIO
import numpy as np
from ultralytics.utils.plotting import Annotator, colors
import json


def draw_annotations(coco_annotations, image_path, confidence_threshold=0.7):
    print("confidence_threshold: ", confidence_threshold)

    image = Image.open(image_path)
    image = np.array(image)

    ann = Annotator(
        image,
        line_width=None,  # default auto-size
        font_size=None,  # default auto-size
        font="Arial.ttf",  # must be ImageFont compatible
        pil=True,
    )

    for object in coco_annotations:
        if object["score"] < confidence_threshold / 100:
            continue
        label = object["category_name"]
        x, y, w, h = object["bbox"]
        box = [x, y, x + w, y + h]
        ann.box_label(box, label, color=colors(object["category_id"], bgr=True))

    return ann.result()


##########
##### Session state management
##########

if "coco_annoataions" not in st.session_state:
    print("<Initialize session state>")
    st.session_state["coco_annoataions"] = []
    st.session_state["image_name"] = []
else:
    print("<Session state already exists>")


##########
##### Set up sidebar.
##########

# Add in location to select image.

st.sidebar.write("### Select an image to upload.")
uploaded_file = st.sidebar.file_uploader(
    "", type=["png", "jpg", "jpeg"], accept_multiple_files=False
)

st.sidebar.write("#")

## Add in sliders.
st.sidebar.write("### Confidence threshold")
confidence_threshold = st.sidebar.slider(
    "What is the minimum acceptable confidence level for displaying a bounding box?",
    value=70,
)


## Display total number for each category
st.sidebar.write("### Total number of each category")
count_for_each_category = {}
for object in st.session_state["coco_annoataions"]:
    if object["score"] < confidence_threshold / 100:
        continue
    if object["category_name"] not in count_for_each_category.keys():
        count_for_each_category[object["category_name"]] = 1
    else:
        count_for_each_category[object["category_name"]] += 1

for key, value in count_for_each_category.items():
    if key == "encarsia":
        key = "encarsia or spidermite"
    st.sidebar.write(key + ": " + str(value))

st.sidebar.write("#")

st.sidebar.write("### Developed by")
image = Image.open("./logo/AAFC_logo.png")
st.sidebar.image(image, use_column_width=True)

image = Image.open("./logo/AC_logo.png")
st.sidebar.image(image, use_column_width=True)

##########
##### Set up main app.
##########

## Title.
st.write("## Arthropod Detection")

## Pull in default image or user-selected image.
if uploaded_file is None:
    # Default image.
    path = "./sample_images"
    img_path = glob.glob(path + "/*.jpg")
    # choose a random image
    img_path = img_path[np.random.randint(0, len(img_path))]

    image = Image.open(img_path)

else:
    # User-selected image.
    image = Image.open(uploaded_file)

## Subtitle.
st.write("### Result")

if uploaded_file is None:

    result_image_path = img_path.replace(
        "sample_images", "sample_result_images"
    ).replace(".jpg", ".png")
    result_image = Image.open(result_image_path)
else:
    if st.session_state["image_name"] != uploaded_file:
        print("\033[91m" + "1. CALLING CLOUD INFERNECE " + "\033[0m")

        # Inference through Modal cloud
        byte_arr = BytesIO()
        image.save(byte_arr, format="PNG")
        byte_arr = byte_arr.getvalue()

        modal_url = "https://jung0072--sahi-inference-predict.modal.run"
        r = requests.post(
            f"{modal_url}",
            data=byte_arr,
        )
        print("\033[91m 2. " + str(r) + "\033[0m")

        coco_annotation = json.loads(r.content)
        print("Total Detections: ", len(coco_annotation))
        st.session_state["coco_annoataions"] = coco_annotation
        st.session_state["image_name"] = uploaded_file

    result_image = draw_annotations(
        st.session_state["coco_annoataions"],
        uploaded_file,
        confidence_threshold=confidence_threshold,
    )


# Display image.
st.image(result_image, use_column_width=True)
