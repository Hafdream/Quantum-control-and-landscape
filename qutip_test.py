from io import BytesIO
import numpy as np
from matplotlib import pyplot as plt
from qutip_qip.circuit import QubitCircuit, Gate
from PIL import Image, ImageOps
from math import pi


def draw_circuit(gate):
    q = QubitCircuit(2, reverse_states=False)
    q.add_gate(gate, controls=[0], targets=[1])
    image_ = q.png
    img_bytesio = BytesIO(image_.data)
    img_pil = Image.open(img_bytesio)
    img_array = np.array(img_pil)
    img_array = img_array[:, :, -1]
    # plt.figure(figsize=(2, 2))
    plt.imshow(img_array, cmap="gray")

    plt.show()


def draw_cricuit2():
    fig, axs = plt.subplots(4, 1)
    N = 3
    qc1 = QubitCircuit(N)
    qc1.add_gate("CNOT", 0, 1)
    qc1.add_gate("CNOT", 1, 0)
    qc1.add_gate("CNOT", 0, 1)
    # latex_code = qc1.latex_code()
    # print(latex_code)

    image1 = qc1.png
    try:
        img_bytesio1 = BytesIO(image1.data)
    except:
        img_bytesio1 = BytesIO(image1)
    img_pil1 = Image.open(img_bytesio1)
    # img_array1 = 1 - (np.array(img_pil1)/255)
    img_array1 = np.array(img_pil1)
    img_array1 = img_array1[:, :, -1]
    # img_array1_1 = np.zeros(img_array1[:, :, 0].shape)
    # img_array1_1 = np.expand_dims(img_array1_1, axis=-1)
    # print(img_array1.shape, img_array1_1.shape)
    # img_array1 = np.concatenate((img_array1, img_array1_1), axis=-1)
    # plt.figure(figsize=(2, 2))
    axs[1].imshow(img_array1, cmap="gray")

    qc3 = QubitCircuit(3)
    qc3.add_gate("CNOT", 1, 0)
    qc3.add_gate("RX", 0, None, pi / 2, r"\pi/2")
    qc3.add_gate("RY", 1, None, pi / 2, r"\pi/2")
    qc3.add_gate("RZ", 2, None, pi / 2, r"\pi/2")
    qc3.add_gate("ISWAP", [1, 2])

    image3 = qc3.png
    try:
        img_bytesio3 = BytesIO(image3.data)
    except:
        img_bytesio3 = BytesIO(image3)
    img_pil3 = Image.open(img_bytesio3)
    # img_array3 = 1 - (np.array(img_pil3)/255)
    img_array3 = np.array(img_pil3)
    img_array3 = img_array3[:, :, -1]
    axs[2].imshow(img_array3, cmap="gray")

    qc5 = qc3.resolve_gates("ISWAP")
    image0 = qc5.png

    try:
        img_bytesio0 = BytesIO(image0.data)
    except:
        img_bytesio0 = BytesIO(image0)
    img_pil0 = Image.open(img_bytesio0)
    # img_array0 = 1 - (np.array(img_pil0)/255)
    img_array0 = np.array(img_pil0)
    img_array0 = img_array0[:, :, -1]
    axs[0].imshow(img_array0, cmap="gray")

    qc4 = qc3.resolve_gates("CNOT")
    image4 = qc4.png
    try:
        img_bytesio4 = BytesIO(image4.data)
    except:
        img_bytesio4 = BytesIO(image4)
    img_pil4 = Image.open(img_bytesio4)
    # img_array4 = 1 - (np.array(img_pil4)/255)
    img_array4 = np.array(img_pil4)
    img_array4 = img_array4[:, :, -1]
    axs[3].imshow(img_array4, cmap="gray")
    plt.show()


if __name__ == "__main__":
    gate_ = "CNOT"
    # draw_circuit(gate_)
    draw_cricuit2()
