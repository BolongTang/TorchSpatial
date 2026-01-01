import torch
import torch.nn as nn

class Tester(nn.Module):
    def __init__(self, dataloader, model): 
        super().__init__()
        self.dataloader = dataloader
        self.model = model

    def test(self):
        total = 0
        correct = 0
        
        for data in self.dataloader:
            inputs, labels = data
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the network on the {total} test images: {100 * correct / total:.2f}%")
            
