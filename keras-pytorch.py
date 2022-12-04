# KERAS
input_3d = (1, 64, 96, 96)
pool_3d = (2, 2, 2)
model = Sequential()
model.add(Convolution3D(8, 3, 3, 3, name="conv1", input_shape=input_3d, data_format="channels_first"))
model.add(MaxPooling3D(pool_size=pool_3d, name="pool1"))
model.add(Convolution3D(8, 3, 3, 3, name="conv2", data_format="channels_first"))
model.add(MaxPooling3D(pool_size=pool_3d, name="pool2"))
model.add(Convolution3D(8, 3, 3, 3, name="conv3", data_format="channels_first"))
model.add(MaxPooling3D(pool_size=pool_3d, name="pool3"))
model.add(Flatten())
model.add(Dense(2000, activation="relu", name="dense1"))
model.add(Dropout(0.5, name="dropout1"))
model.add(Dense(500, activation="relu", name="dense2"))
model.add(Dropout(0.5, name="dropout2"))
model.add(Dense(3, activation="softmax", name="softmax"))

# PYTORCH
class CNN(nn.Module):
    def __init__(
        self,
    ):
        super(CNN, self).__init__()

        self.maxpool = nn.MaxPool3d((2, 2, 2))

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3)
        self.conv3 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3)

        self.linear1 = nn.Linear(4800, 2000)
        self.dropout1 = nn.Dropout3d(0.5)

        self.linear2 = nn.Linear(2000, 500)
        self.dropout2 = nn.Dropout3d(0.5)

        self.linear3 = nn.Linear(500, 3)

    def forward(self, x):

        out = self.maxpool(self.conv1(x))
        out = self.maxpool(self.conv2(out))
        out = self.maxpool(self.conv3(out))

        # Flattening process
        b, c, d, h, w = out.size()  # batch_size, channels, depth, height, width
        out = out.view(-1, c * d * h * w)

        out = self.dropout1(self.linear1(out))
        out = self.dropout2(self.linear2(out))
        out = self.linear3(out)

        out = torch.softmax(out, 1)

        return out
