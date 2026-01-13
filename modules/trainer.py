import torch
import numpy as np

# Some models like the included location encoders only supports list or np.ndarray
# Coerce datatype from torch.Tensor to np.ndarray briefly, then turn it back after processing
def forward_with_np_array(batch_data, model):
    loc_b = np.array(batch_data)
    loc_b = np.expand_dims(batch_data, axis=1)
    loc_embedding = torch.squeeze(model(coords = loc_b))
    return loc_embedding

def train(epochs, 
            batch_count_print_avg_loss,
            dataloader,
            loc_encoder,
            model,
            criterion,
            optimizer, 
            device):
    
    model = model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            img_b, loc_b, y_b = data

            # assume loc_b have [lat, long]
            img_embedding = img_b
            loc_embedding = forward_with_np_array(batch_data = loc_b, model = loc_encoder)

            loc_img_interaction_embedding = torch.mul(loc_embedding, img_embedding)

            optimizer.zero_grad()
            outputs = torch.nn.Softmax(dim=1)(loc_img_interaction_embedding)
            
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % batch_count_print_avg_loss == batch_count_print_avg_loss - 1:
                print('[epoch %d, batch %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / batch_count_print_avg_loss))

                running_loss = 0.0

    print(f'Training Completed.')