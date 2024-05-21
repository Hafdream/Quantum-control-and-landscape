import gradio
from PIL import Image
import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
import skorch
import gradio

# Set some options so Jupyter displays more of Pandas DataFrames than default behaviour.
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_colwidth', 200)


def load_data(file_name):

    # file_name = '../data/mnist_train.csv'
    target_feature = "label"
    num_classes = 10
    classes = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9"}
    dataset = pd.read_csv(file_name)
    X = dataset.drop([target_feature], axis=1)
    y = dataset[target_feature]
    X = X.astype('float32')
    y = y.astype('int64')
    # X = X/255
    X = X * 2 - 1
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=0,
                                                                                train_size=0.7, stratify=y)
    return x_train, y_train


class MyMnistMlp(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.dense0 = torch.nn.Linear(784, 512)
        self.nonlin0 = torch.nn.ReLU()
        self.dense1 = torch.nn.Linear(512, 256)
        self.nonlin1 = torch.nn.ReLU()
        self.output = torch.nn.Linear(256, 10)

    def forward(self, X, **kwargs):
        X = self.nonlin0(self.dense0(X))
        X = self.nonlin1(self.dense1(X))
        X = self.output(X)
        X = torch.nn.functional.softmax(X, dim=-1)
        return X


def recognize_digit(image):
    image = image["composite"]
    # image = Image.fromarray((np.array(image)).astype('uint8')).convert('L').resize((280, 280))
    print(image.getextrema())

    image = image.resize((28, 28))
    # image.show()
    image = np.asarray(image)
    image = image.reshape(1, 784)
    image = image.astype('float32')
    image = image / 255
    image = image * 2 - 1

    prediction = net.predict_proba(image)
    prediction = {str(i): float(prediction[0][i]) for i in range(10)}
    return prediction


if __name__ == "__main__":
    X_train, Y_train = load_data(
        "/Users/haf/Library/CloudStorage/GoogleDrive-haftu.fentaw@ucdconnect.ie/My Drive/COMP47590_Labs/mnist_train.csv")

    net = skorch.NeuralNetClassifier(MyMnistMlp, max_epochs=150, lr=0.01, iterator_train__shuffle=True)
    net.fit(X_train.values, Y_train.values)
    im = gradio.ImageEditor(image_mode='L', type='pil')
    label = gradio.Label(num_top_classes=3)

    interface = gradio.Interface(fn=recognize_digit, inputs=im, outputs=label, live=True)
    interface.launch()
