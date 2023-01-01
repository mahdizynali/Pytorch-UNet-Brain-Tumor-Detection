from config import *
from plotter import *
from trainer import *
from brain import *

'''

Before running the program, you have to make sure that every moduls are installed
and change program's directories to read and save data

'''
#==========================================================================
# set a seeds to use for some probable randomly works

def set_seed(seed = 0):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
set_seed()

def main ():
    
    #==============================================================
    # const parameters
    dataSets = "/home/mahdi/Desktop/project/dataset/kaggle_3m/*"
    # File path line length images for later sorting
    BASE_LEN = 89 # len(TCGA_DU_6404_19850629_ <-!!!43.tif)
    END_IMG_LEN = 4 # len(TCGA_DU_6404_19850629_43 !!!->.tif)
    END_MASK_LEN = 9 # len(TCGA_DU_6404_19850629_43 !!!->_mask.tif)
    
    #==============================================================
     # in this case we try to extract the path of data sets
    print("\n\nGathering dataSets...\n") 
    addresses = []
    for path in glob.glob(dataSets):
        try:
            directoryName = path.split("/")[-1] # get name of files
            for imgName in os.listdir(path):
                image_path = (path + "/" + imgName)
                addresses.extend([directoryName, image_path])
        except:
            print("Trouble to find directory !!\n\n")
    #==============================================================   
        
    # now we creat a primier dataFrame with pandas and try to arange data 
    df = pd.DataFrame({"directory" : addresses[::2], "path" : addresses[1::2]})

    # spliting images from masks
    df_images = df[~df["path"].str.contains("mask")]
    df_masks = df[df["path"].str.contains("mask")]

    # sorting each data set to specific mask
    images = sorted(df_images["path"].values, key=lambda x : (x[BASE_LEN:-END_IMG_LEN]))
    masks = sorted(df_masks["path"].values, key=lambda x : (x[BASE_LEN:-END_MASK_LEN]))

    # After sorting data, now we creat final dataFrame
    df = pd.DataFrame({"directory": df_images["directory"], "image_path": images, "mask_path": masks})

    #==============================================================
    # spliting data if there is any diagnosted mask or not 
    def maskDiagnosis(mask_path):
        value = np.max(cv2.imread(mask_path)) # color black = 0 & color white = 255
        if value > 0 :
            return 1 # there is a mask of shape
        else: 
            return 0 # there is not 
        
    df["status"] = df["mask_path"].apply(lambda status: maskDiagnosis(status))
    df.to_csv("/home/mahdi/Desktop/project/dataFrame.csv") # save data frame

    # plotting result 
    # Plotter(df["status"].value_counts(), len(df)).plot_Status()
    # Plotter(df).plot_Status_Distribution()

    #==============================================================
    # random visulizing data by sendeing count number
    # Plotter(df).random_data_visualize(5)

    # random visulizing positive cases by sendeing count number
    # Plotter(df).random_positive_patient(5)
    
    #==============================================================
    # now we split datasets into (Train - Validation - Test) in a radom form
    train, validation = train_test_split(df, stratify=df['status'], test_size=0.1, random_state=0)
    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)

    train, test = train_test_split(train, stratify=train['status'], test_size=0.15, random_state=0)
    train = train.reset_index(drop=True)
    
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

    # def dataset_info(dataset): 
    #     print(f'Size of dataset: {len(dataset)}')
    #     index = random.randint(1, 40)
    #     img, label = dataset[index]
    #     print(f'Sample-{index} Image size: {img.shape}, Mask: {label.shape}\n')

    # print('Train dataset:')
    # dataset_info(train_ds)
    # print('Validation dataset:')
    # dataset_info(val_ds)
    # print('Test dataset:')
    # dataset_info(test_ds)
    
    #==============================================================
    # in this case we make a frame of data loader for every dataset
    # actually we set the datasets into the DataLoader class which is abstraced for BrainDatasets class
    
    batch_size = 64

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)  
    val_dl = DataLoader(val_ds, batch_size, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size, num_workers=2, pin_memory=True)

    # images, masks = next(iter(train_dl)) 
    # print(images.shape)
    # print(masks.shape)

    #==============================================================
    # now creating a model and set it to the device (cpu or gpu)
    
    print("Creating model ...\n")
    model = UNet(3, 1).to(device)
    out = model(torch.randn(1, 3, 128, 128).to(device))
    print(f"Model size : {out.shape}\n\n")

    #==============================================================
    # let's train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
    num_epochs = 30 # number of training duration
    
    # (train_loss_history , 
    #  train_dice_history , 
    #  val_loss_history   , 
    #  val_dice_history   ) = train_model(device, model, train_dl, val_dl, bce_dice_loss, 
    #                                     optimizer, scheduler, num_epochs) 
    
    #==============================================================
    
    # Plotter.plot_dice_history('UNET', train_dice_history, val_dice_history, num_epochs)
    # Plotter.plot_loss_history('UNET', train_loss_history, val_loss_history, num_epochs)

    # torch.save(model.state_dict(), 'model.pth')
    
    #==============================================================
    # load trained model and test prediction
    
    model.load_state_dict(torch.load('model.pth'))
    
    test_dice, test_loss = eval_loop(device, model, test_dl, bce_dice_loss, scheduler, training=False)
    print("Mean IoU/DICE: {:.3f}%, Loss: {:.3f}\n\n".format((100*test_dice), test_loss))
    
    # set a number instead of 19
    Plotter.plot_test_prediction(model, device, test, 19)

    #==============================================================
    
if __name__ == '__main__' :
    main()
else :
    print("You are not administrator !\n\n")