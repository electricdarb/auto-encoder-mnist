import torch 
import torch.nn.functional as F
from commons.load_data import load_data
from tqdm import tqdm

INPUT_DIMS = (28, 28)

class AutoEnc(torch.nn.Module):
    def __init__(self) -> None:
        super(AutoEnc, self).__init__()
        # encoder
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(INPUT_DIMS[0] * INPUT_DIMS[1], 128)
        self.linear2 = torch.nn.Linear(128, 32)
        self.linear3 = torch.nn.Linear(32, 10)

        # decoder 
        self.linear4 = torch.nn.Linear(10, 32)
        self.linear5 = torch.nn.Linear(32, 128)
        self.linear6 = torch.nn.Linear(128, INPUT_DIMS[0] * INPUT_DIMS[1])


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # encoder
        x = self.flatten(input)

        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)

        # decoder 
        x = self.linear4(x)
        x = F.relu(x)
        x = self.linear5(x)
        x = F.relu(x)
        x = self.linear6(x)
        x = torch.sigmoid(x)

        x = torch.reshape(x, (-1, 1, *INPUT_DIMS))

        return x

def train(model = None):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_data, test_data = load_data() # load in data

    epochs = 100
    batch_size = 32

    train = torch.utils.data.DataLoader(train_data, batch_size = batch_size)
    test = torch.utils.data.DataLoader(test_data, batch_size = batch_size)

    loss_fn = torch.nn.BCELoss()

    model = model if model else AutoEnc()

    opt = torch.optim.SGD(model.parameters(), lr = .001)

    model.to(device)

    print('Start Training')
    for epoch in range(epochs):
        train_loss = 0 # init train loss to 0

        for imgs, _ in tqdm(train):
            imgs = imgs.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, imgs)

            loss.backward()
            opt.step()
            train_loss += loss.item() 

        train_loss /= len(train)
        
        print(f"Epoch: {epoch}/{epochs}\tTrain loss: {train_loss:5f}")
    
    return model


if __name__ == "__main__":
    train()








