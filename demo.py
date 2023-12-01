import streamlit as st
import torch
import cv2
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
from torchvision import transforms
import base64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TwoStream3DConvNet(nn.Module):
    def __init__(self, num_classes):
        super(TwoStream3DConvNet, self).__init__()
        # """
        # A two-stream 3D Convolutional Neural Network for video classification.
        # This network processes spatial and temporal information separately and then combines them.

        # :param num_classes (int): Number of classes for classification.
        # """

        # Spatial Stream
        self.spatial_stream = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        
        # Temporal Stream
        self.temporal_stream = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(16384, 512)
        self.fc2 = nn.Linear(512, num_classes)
        # self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the network.

        :param x (Tensor): Input tensor of shape (batch_size, 3, max_length, 112, 112)
        :return: Output tensor of shape (batch_size, num_classes)
        """
        spatial_out = self.spatial_stream(x)
        temporal_out = self.temporal_stream(x)
        
        # Concatenate the outputs of the two streams
        combined = torch.cat((spatial_out, temporal_out), dim=1)
        
        combined = torch.flatten(combined, 1)
        # combined = self.dropout(combined)
        combined = self.relu(self.fc1(combined))
        # combined = self.dropout(combined)
        combined = self.fc2(combined)
        return combined

    
def crop_frame(frame):
    """
    Crops the frame to a square shape
    :param frame: frame to crop
    :return: cropped frame
    """
    height, width = frame.shape[:2]
    th_dim = frame.shape[2]
    max_dim = max(height, width)
    dif = abs(height-width)

    first_side = dif // 2
    second_side = dif - first_side
    
    
    if width == max_dim:
        f_array = np.zeros(shape=(first_side, max_dim, th_dim))
        s_array = np.zeros(shape=(second_side, max_dim, th_dim))
        frame = np.concatenate((f_array, np.array(frame), s_array), axis=0)
    else:
        f_array = np.zeros(shape=(max_dim, first_side, th_dim))
        s_array = np.zeros(shape=(max_dim, second_side, th_dim))
        frame = np.concatenate((f_array, np.array(frame), s_array), axis=1)

    return frame

def load_video(path, img_size, num_frames=132):
    """
    Loads the video from the path and returns a tensor of frames
    :param path: path to the video
    :param img_size: size of the image
    :param num_frames: number of frames to sample
    :return: tensor of frames
    """
    cap = cv2.VideoCapture(path)
    frames = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % 4 == 0:
            frame = crop_frame(frame)
            frame = cv2.resize(frame, (img_size, img_size))
            frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
            frame_tensor = torch.Tensor(frame).permute(2, 0, 1).to(device)
            frames.append(frame_tensor)
        i += 1
        
    return frames

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PaddedSignLanguageDataset(Dataset):
    def __init__(self, annotations, transform=None, max_length=None):
        """
        Custom dataset for loading sign language video tensors with padding.
        Each video tensor is padded to a uniform length for consistent processing.

        :param annotations (DataFrame): DataFrame containing the annotations.
        :param transform (callable, optional): Optional transform to be applied on a sample.
        :param max_length (int, optional): Maximum length of the video tensors. If not provided, it will be calculated.
        """
        self.annotations = annotations
        self.transform = transform
        self.max_length = 64
        self.tensor_path = ""

        if self.max_length is None:
            # Calculate the maximum length among all tensors
            self.max_length = max(len(self.tensor_path + torch.load(row['attachment_id'], map_location=torch.device('cpu'))) for _, row in annotations.iterrows())

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Returns the sample at the given index.

        :param idx (int): Index
        :return: Tuple of (video tensor, label)
        """
        tensor_path = self.annotations.iloc[idx]['attachment_id']
        label = self.annotations.iloc[idx]['text']
        
        # Load the tensor
        tensor = torch.load(self.tensor_path + tensor_path, map_location=torch.device('cpu'))

        # Pad the tensor to the maximum length
        padded_tensor = torch.zeros((self.max_length, *tensor[0].shape))
        padded_tensor[:len(tensor)] = torch.stack(tensor)
                
        # Apply transform if any
        if self.transform:
            padded_tensor = self.transform(padded_tensor)

        return padded_tensor, label
    
model = TwoStream3DConvNet(num_classes=5)
model_path = "two_stream_3d_conv_net_val48.pt"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Streamlit interface
st.title("Sign Language Classification")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])

if uploaded_file is not None:

    file_bytes = uploaded_file.read()
    base64_bytes = base64.b64encode(file_bytes).decode()
    base64_url = f"data:video/mp4;base64,{base64_bytes}"

    # HTML string with video tag
    video_html = f"""
        <video width="320" height="240" controls autoplay>
            <source src="{base64_url}" type="video/mp4">
        </video>
    """

    # Display the video using HTML
    st.markdown(video_html, unsafe_allow_html=True)
    # Process the uploaded video file
    with st.spinner('Processing...'):
        temp_file = Path(uploaded_file.name)
        temp_file.write_bytes(file_bytes)

        frames = load_video(str(temp_file), 64, 64)

        # print(frames)

        tensor = torch.stack(frames)
        # print(tensor.shape)

        # Convert frames to tensor and add necessary dimensions
        padded_tensor = torch.zeros((64, *tensor[0].shape))
        # print(padded_tensor.shape)
        padded_tensor[:len(tensor)] = tensor
        # print(padded_tensor.shape)
        padded_tensor = padded_tensor.reshape((1, 3, 64, 64, 64)) # Adjust dimensions as required by the model
        # print(padded_tensor.shape)

        # Prediction
        with torch.no_grad():
            output = model(padded_tensor)
            pred_class = torch.argmax(output, dim=1)
            label_map = {'MakDonalds': 0, 'Добро пожаловать!': 1, 'Пока': 2, 'Привет!': 3, 'С днем рождения': 4}
            rev_label_map = {v: k for k, v in label_map.items()}
            st.write("Predicted Class:", rev_label_map[pred_class.item()])

    # Clean up
    temp_file.unlink()

st.sidebar.write("Upload a video to get started")