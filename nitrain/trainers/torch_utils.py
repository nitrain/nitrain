import numpy as np


def torch_model_fit(model, loss, optimizer, metrics, device, loader, epochs, validation, **kwargs):
    """
    Fit a torch model on a loader
    """
    import torch
    
    # start training loop
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for inputs, outputs in loader:
            step += 1
            # TODO: support multiple inputs + outputs here
            inputs = torch.tensor(inputs).to(device)
            labels = torch.tensor(outputs).to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss_value = loss(outputs, labels)
            loss_value.backward()
            optimizer.step()
            
            epoch_loss += loss_value.item()
            
            print(f"{step}/{len(loader.dataset) // loader.images_per_batch}, " f"train_loss: {loss_value.item():.4f}")
            epoch_len = len(loader.dataset) // loader.images_per_batch
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        # validation
        if validation:
            results = torch_model_evaluate(model, metrics, device, validation)
            result = results[0]
            acc_metric = results[1]
            metric_values.append((result, acc_metric))
            if result > best_metric:
                best_metric = result
                best_metric_epoch = epoch + 1
                #torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current AUC: {result:.4f}"
                f" current accuracy: {acc_metric:.4f}"
                f" best AUC: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
    return metric_values
   

def torch_model_predict():
    pass


def torch_model_evaluate(model, metrics, device, loader):
    import torch
    metric_values = []
    model.eval()
    with torch.no_grad():
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)
        for val_data in loader:
            val_images, val_labels = (
                torch.Tensor(val_data[0]).to(device),
                torch.Tensor(val_data[1]).long().to(device),
            )
            y_pred = torch.cat([y_pred, model(val_images)], dim=0)
            y = torch.cat([y, val_labels], dim=0)

            for metric_fn in metrics:
                result = metric_fn(y, y_pred)
                metric_values.append(result)
    
    return metric_values
