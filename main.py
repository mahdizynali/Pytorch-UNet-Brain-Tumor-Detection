# Authored By Mahdi Zeinali
# github : github.com/mahdizynali

from config import *
from plotter import *
from trainer import *
from brain import *

'''
Before running the program, you have to make sure that every moduls are installed
and change program's directories to read and save data
'''

def trainNewModel (model, optimizer, scheduler):
    '''train a new model'''
    return train_model(device, model, train_dl, val_dl, bce_dice_loss, optimizer, scheduler) 
    
def loadLastModel (model, scheduler):
    '''load trained model and test prediction'''
    model.load_state_dict(torch.load(file_path + 'model.pth')) 
    # model.load_state_dict(torch.load('model.pth',map_location=torch.device('cpu'))) # to use with cpu only
    return eval_loop(device, model, test_dl, bce_dice_loss, scheduler, training=False)

def plotting ():
    '''displaying data and results'''
    
    print("choose a section : \n")
    print('''
    1 : display patients tumor status\n
    2 : display tumor status distributon\n
    3 : display random image of samples\n
    4 : display random image of positive case\n
    5 : display random machine prediction\n
    0 : exit''')

    while(True) :
        state = input()
        
        if state == ('1') :
            Plotter(df["status"].value_counts(), len(df)).plot_Status()
            
        elif state == ('2'):
            Plotter(df).plot_Status_Distribution()
        
        elif state == ('3'):
            Plotter(df).random_data_visualize(5)
            
        elif state == ('4'):
            Plotter(df).random_positive_patient(5)
            
        elif state == ('5'):
            Plotter.plot_test_prediction(model, device, test, 5)  
            
        else : exit(0)

#========================================================================== 
 # in this case we try to extract the path of data sets
print("\n\nGathering dataSets...\n") 

mask_files = glob.glob(file_path + "dataSets/" + '*/*_mask*')
image_files = [file.replace('_mask', '') for file in mask_files]

def diagnosis(mask_path):
    return 1 if np.max(cv2.imread(mask_path)) > 0 else 0

df = pd.DataFrame({"image_path": image_files,
                  "mask_path": mask_files,
                  "status": [diagnosis(x) for x in mask_files]})

df.to_csv(file_path + "dataFrame.csv") # save data frame

#==============================================================
# now we split datasets into (Train - Validation - Test) in a radom form
train, validation = train_test_split(df, stratify=df['status'], test_size=0.1, random_state=0)
train = train.reset_index(drop=True)
validation = validation.reset_index(drop=True)

train, test = train_test_split(train, stratify=train['status'], test_size=0.15, random_state=0)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

print(f"Train : {train.shape}\nValidation : {validation.shape}\nTest : {test.shape}\n\n")

#==============================================================
# in this case we try to fix every dataset in size and angle like each other 
train_transform = A.Compose([
    A.Resize(width=128, height=128, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
])
validation_transform = A.Compose([
    A.Resize(width=128, height=128, p=1.0),
    A.HorizontalFlip(p=0.5),
])
test_transform = A.Compose([
    A.Resize(width=128, height=128, p=1.0)
])
set_seed()
train_ds = BrainDataset(train, train_transform)
val_ds = BrainDataset(validation, validation_transform)
test_ds = BrainDataset(test, test_transform)

#==============================================================
# in this case we split the actual data and test part for testing after training

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)  
val_dl = DataLoader(val_ds, batch_size, num_workers=2, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size, num_workers=2, pin_memory=True)

#==============================================================
# now creating a model and set it to the device (cpu or gpu)

print("Creating a model ...\n")
model = UNet(3, 1).to(device)
out = model(torch.randn(1, 3, 128, 128).to(device))
print(f"Model size : {out.shape}\n\n")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3) 

#==============================================================

print("\nchoose an option : \n\n")
print('''1 : train a new model \n2 : use the last model\n0 : exit\n''')

while(True) :
    state = input()
    
    if state == ('1') :
        (train_loss_history, train_dice_history, val_loss_history, val_dice_history) = trainNewModel(model, optimizer, scheduler)
        torch.save(model.state_dict(), file_path + 'model.pth')
        print("\n\n\nAccuracy of trained model is : ", train_dice_history * 100 , " %\n\n")
        Plotter.plot_dice_history('UNET', train_dice_history, val_dice_history, num_epochs)
        Plotter.plot_loss_history('UNET', train_loss_history, val_loss_history, num_epochs)
        plotting()
        
    elif state == ('2'):
        print("\ntesting data on previous model ...\n")
        test_dice, test_loss = loadLastModel(model, scheduler)
        print(f"\nMean IoU/DICE: {(100*test_dice):.3f}%, Loss: {test_loss:.3f}\n\n")
        plotting()
    
    else : exit(0)
