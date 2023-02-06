from config import *


def dice_coef_metric(pred, label):
    intersection = 2.0 * (pred * label).sum()
    union = pred.sum() + label.sum()
    if pred.sum() == 0 and label.sum() == 0:
        return 1.
    return intersection / union

def dice_coef_loss(pred, label):
    smooth = 1.0
    intersection = 2.0 * (pred * label).sum() + smooth
    union = pred.sum() + label.sum() + smooth
    return 1 - (intersection / union)

def bce_dice_loss(pred, label):
    dice_loss = dice_coef_loss(pred, label)
    bce_loss = nn.BCELoss()(pred, label)
    return dice_loss + bce_loss


def train_loop(device ,model , loader, loss_func, optimizer):
    model.train()
    train_losses = []
    train_dices = []
    
    for i, (image, mask) in enumerate(tqdm(loader)):
    #for i, (image, mask) in enumerate(loader):
        image = image.to(device)
        mask = mask.to(device)
        outputs = model(image)
        out_cut = np.copy(outputs.data.cpu().numpy())
        out_cut[np.nonzero(out_cut < 0.5)] = 0.0
        out_cut[np.nonzero(out_cut >= 0.5)] = 1.0            

        dice = dice_coef_metric(out_cut, mask.data.cpu().numpy())
        loss = loss_func(outputs, mask)
        train_losses.append(loss.item())
        train_dices.append(dice)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    return train_dices, train_losses


def eval_loop(device, model, loader, loss_func, scheduler, training=True):
    model.eval()
    val_loss = 0
    val_dice = 0

    with torch.no_grad():
        for step, (image, mask) in enumerate(loader):
            image = image.to(device)
            mask = mask.to(device)
    
            outputs = model(image)
            loss = loss_func(outputs, mask).data.cpu().numpy()
            
            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < 0.5)] = 0.0
            out_cut[np.nonzero(out_cut >= 0.5)] = 1.0
            dice = dice_coef_metric(out_cut, mask.data.cpu().numpy())

            #accuracy
 
            val_loss += loss
            val_dice += dice
        
        val_mean_dice = val_dice / step
        val_mean_loss = val_loss / step
        
        if training:
            scheduler.step(val_mean_dice)
        
    return val_mean_dice, val_mean_loss


def train_model(device, model, train_loader, val_loader, loss_func, optimizer, scheduler):
    train_loss_history = []
    train_dice_history = []
    val_loss_history = []
    val_dice_history = []
    
    for epoch in range(num_epochs):
        train_dices, train_losses = train_loop(device, model, train_loader, loss_func, optimizer)
        train_mean_dice = np.array(train_dices).mean()
        train_mean_loss = np.array(train_losses).mean()
        val_mean_dice, val_mean_loss = eval_loop(device, model, val_loader, loss_func, scheduler)
        
        train_loss_history.append(np.array(train_losses).mean())
        train_dice_history.append(np.array(train_dices).mean())
        val_loss_history.append(val_mean_loss)
        val_dice_history.append(val_mean_dice)

        print(f'\nEpoch step: {epoch+1}/{num_epochs}\nTrain Loss: {train_mean_loss:.3f}\nValidation Loss: {val_mean_loss:.3f}')
        print(f'Train DICE: {train_mean_dice:.3f}\nValidation DICE: {val_mean_dice:.3f}\n')
        print("="*50)
        
    return train_loss_history, train_dice_history, val_loss_history, val_dice_history
