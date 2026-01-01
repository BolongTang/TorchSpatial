from torch import nn


class Trainer(nn.Module):
    def __init__(self, dataloader, model, criterion, optimizer): # criterion is loss
        super().__init__()
        self.dataloader = dataloader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
    
    def train(self, epochs, batch_count_print_avg_loss = 2000):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(self.dataloader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                if i % batch_count_print_avg_loss == batch_count_print_avg_loss - 1:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / batch_count_print_avg_loss))

                    running_loss = 0.0