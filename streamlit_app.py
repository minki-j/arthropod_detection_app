import streamlit as st
import requests
from PIL import Image
import glob
from io import BytesIO
import numpy as np
from ultralytics.utils.plotting import Annotator, colors
import json
import numpy as np
import pandas as pd


def bbox_iou(box, boxes, iou_thres, starting_index):
    indices = []
    for i, b in enumerate(boxes):
        x1_1, y1_1, w1, h1 = box
        x1_2, y1_2, w2, h2 = b
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2

        # Calculate area of individual bounding boxes
        area_bbox1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area_bbox2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        # Calculate intersection coordinates
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        # Calculate area of intersection rectangle
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        # Calculate union area
        union_area = area_bbox1 + area_bbox2 - inter_area

        # Calculate IoU
        iou = inter_area / union_area if union_area > 0 else 0
        if iou > iou_thres:
            indices.append(starting_index + i + 1)

    return indices


def remove_overlap(coco_annotations):
    # pick the best score if iou is over 0.5
    coco_annotations = sorted(coco_annotations, key=lambda x: x["score"], reverse=True)

    object_indices_to_remove = []

    for i, object in enumerate(coco_annotations):
        if i in object_indices_to_remove:
            continue

        boxes_to_compare = [
            object["bbox"]
            for j, object in enumerate(coco_annotations)
            if i < j
        ]

        if len(boxes_to_compare) == 0:
            continue

        iou = bbox_iou(
            object["bbox"],
            boxes_to_compare,
            iou_thres=0.3,
            starting_index=i,
        )

        object_indices_to_remove.extend(iou)

    coco_annotations = [
        object
        for i, object in enumerate(coco_annotations)
        if i not in object_indices_to_remove
    ]

    return coco_annotations


def draw_annotations(coco_annotations, image_path, confidence_threshold=0.7):

    abbreviations = {
        "encarsia": "EN / SP",
        "melon_aphid": "MA",
        "nesidiocoris_tenuis": "NT",
        "orius_insidiosus": "OI",
        "western_flower_thrips": "WFT",
        "white_fly": "WF",
    }
    image = Image.open(image_path)
    image = np.array(image)

    ann = Annotator(
        image,
        line_width=None,  # default auto-size
        font_size=50,  # default auto-size
        font="Arial.ttf",  # must be ImageFont compatible
        pil=True,
    )

    for object in coco_annotations:
        if object["score"] < confidence_threshold / 100:
            continue
        label = object["category_name"]
        if label == "others":
            continue
        label = abbreviations[label]
        label_and_score = f"{label} {object['score']:.2f}"
        x, y, w, h = object["bbox"]
        box = [x, y, x + w, y + h]
        ann.box_label(box, label_and_score, color=colors(object["category_id"], bgr=True))

    return ann.result()


##########
##### Session state management
##########

if "coco_annoataions" not in st.session_state:
    print("<Initialize session state>")
    st.session_state["coco_annoataions"] = []
    st.session_state["image_path"] = []
    st.session_state["confidence_threshold"] = 0.7
    st.session_state["default_image_path"] = []
    st.session_state["coco_annoataions_overlap_removed"] = []


##########
##### Set up sidebar.
##########
def show_another_example():
    st.session_state["image_path"] = []
    st.session_state["coco_annoataions_overlap_removed"] == []

st.sidebar.button("Show another example", on_click=show_another_example)

uploaded_file = st.sidebar.file_uploader(
    "Select an image to upload",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
)

## Pull in default image
if uploaded_file is None and (
    st.session_state["image_path"] != st.session_state["default_image_path"]
    or st.session_state["image_path"] == []
):
    # Default image.
    path = "./sample_images"
    default_img_paths = glob.glob(path + "/*.jpg")
    # choose a random image
    st.session_state["default_image_path"] = default_img_paths[
        np.random.randint(0, len(default_img_paths))
    ]


st.sidebar.write("#")


## Add in sliders.
st.sidebar.write("### Confidence threshold")
confidence_threshold = st.sidebar.slider(
    "What is the minimum acceptable confidence level for displaying a bounding box?",
    value=70,
)

is_overlap_removed = st.sidebar.checkbox("Remove Overlapping Bounding Boxes")

## get prediction result
if uploaded_file is None:
    if st.session_state["image_path"] != st.session_state["default_image_path"]:
        coco_array_path = st.session_state["default_image_path"].replace(
            ".jpg", ".pickle"
        )
        # use pickel to load array
        with open(coco_array_path, "rb") as f:
            coco_array = np.load(f, allow_pickle=True)
            st.session_state["coco_annoataions"] = coco_array
        st.session_state["image_path"] = st.session_state["default_image_path"]
else:
    if st.session_state["image_path"] != uploaded_file:
        print("\033[91m" + "CALLING CLOUD INFERNECE " + "\033[0m")

        # Create placeholders
        text_placeholder = st.empty()
        progress_placeholder = st.empty()

        text_placeholder.write(
            'AI is detecting arthropods in your image. On average, it takes about 30 seconds.'
        )
        progress_bar = progress_placeholder.progress(0)

        # Inference through Modal cloud
        byte_arr = BytesIO()
        image = Image.open(uploaded_file)
        image.save(byte_arr, format="PNG")
        byte_arr = byte_arr.getvalue()

        progress_bar.progress(50)

        modal_url = "https://jung0072--sahi-inference-predict.modal.run"
        response = requests.post(f"{modal_url}", data=byte_arr)

        progress_bar.progress(100)

        coco_annotation = json.loads(response.content)
        st.session_state["coco_annoataions"] = coco_annotation
        st.session_state["image_path"] = uploaded_file

        # Clear placeholders
        text_placeholder.empty()
        progress_placeholder.empty()

## Display total number for each category
st.sidebar.write("### Total number of each category")
count_for_each_category = {
    "encarsia": 0,
    "melon_aphid": 0,
    "nesidiocoris_tenuis": 0,
    "orius_insidiosus": 0,
    "western_flower_thrips": 0,
    "white_fly": 0,
}
for object in st.session_state["coco_annoataions"]:
    if object["score"] < confidence_threshold / 100:
        continue
    if object["category_name"] not in count_for_each_category.keys():
        count_for_each_category[object["category_name"]] = 1
    else:
        count_for_each_category[object["category_name"]] += 1

count_for_each_category["encarsia / spidermite"] = count_for_each_category.pop(
    "encarsia"
)

df = pd.DataFrame(list(count_for_each_category.items()), columns=["Category", "Count"])
st.sidebar.dataframe(df)


st.sidebar.write("### Developed by")
image = Image.open("./logo/AAFC_logo.png")
st.sidebar.image(image, use_column_width=True)

image = Image.open("./logo/AC_logo.png")
st.sidebar.image(image, use_column_width=True,)

##########
##### Set up main app.
##########


# remove overlaps if not removed
if is_overlap_removed or st.session_state["coco_annoataions_overlap_removed"] == []:
    st.session_state["coco_annoataions_overlap_removed"] = remove_overlap(
        st.session_state["coco_annoataions"]
    )

# Display image.
if is_overlap_removed:
    result_image = draw_annotations(
        st.session_state["coco_annoataions_overlap_removed"],
        st.session_state["image_path"],
        confidence_threshold=confidence_threshold,
    )
else:
    result_image = draw_annotations(
        st.session_state["coco_annoataions"],
        st.session_state["image_path"],
        confidence_threshold=confidence_threshold,
    )

st.image(result_image, use_column_width=True)