from config import *

class Plotter:
    '''specific class to plotting this project's data'''
    scale = 512 # pixel
    
    def __init__ (self, frame, *args):
        self.frame = frame
        self.extraArgument = args
        
    #=======================================================================
    def plot_Status (self):
        '''plotting whether tumor is diagnosted or not'''
        
        # set kind of plotter
        axis = self.frame.plot(kind='bar', 
                            stacked=True, figsize=(10, 6), 
                            color=["red", "green"])
        # set dependensies of plotter
        axis.set_xticklabels(["Negative", "Positive"], rotation=45, fontsize=12)
        axis.set_ylabel('Total instance', fontsize = 12)
        axis.set_title("Plotting the result of diagnosted tumor",fontsize = 18, y=1.05)
        # Annotate
        for i, rows in enumerate(self.frame.values):
            axis.annotate(int(rows), xy=(i, rows-12), 
                        rotation=0, color="white", 
                        ha="center", verticalalignment='bottom', 
                        fontsize=15, fontweight="bold")
        # tag 
        axis.text(1.2, 2550, f"Total {self.extraArgument[0]} instance", size=15,
                color="black",
                ha="center", va="center",
                bbox=dict(boxstyle="round", fc=("lightblue"), ec=("black"),))
        
        # plt.show(block=False)
        # plt.pause(10) # uncommand to set automatic closing time
        
        plt.savefig(file_path + "png/status.png", bbox_inches='tight', pad_inches=0.2, transparent=False)# plt.close()
        plt.show()
    #=======================================================================
    
    def plot_Status_Distribution (self) :
        '''plotting distributed tumor's status between existed directories'''
        
        status = self.frame.groupby(['image_path', 'status'])['status'].size().unstack().fillna(0)
        status.columns = ["Positive", "Negative"]
        # Plot
        ax = status.plot(kind='bar',stacked=True,
                                        figsize=(18, 10),
                                        color=["springgreen", "red"], 
                                        alpha=0.9)
        ax.legend(fontsize=20)
        ax.set_xlabel('Patients',fontsize = 20)
        ax.set_ylabel('Total Images', fontsize = 20)
        ax.set_title("Plotting tumor's status distribution between existed directories",fontsize = 25, y=1.005)
        plt.savefig(file_path + "png/distribution.png", bbox_inches='tight', pad_inches=0.2, transparent=False)
        plt.show()
        
    #=======================================================================
    def random_data_visualize (self, sampleSize):
        '''random display datasets'''

        # select random images
        false = self.frame[self.frame["status"] == 1].sample(sampleSize)['image_path'].values
        true = self.frame[self.frame["status"] == 0].sample(sampleSize)['image_path'].values
        
        # resize and fit the size of images
        imgs = []
        for i, (yes, no) in enumerate(zip(false, true)):
            no = cv2.resize(cv2.imread(no), (self.scale, self.scale))
            yes = cv2.resize(cv2.imread(yes), (self.scale, self.scale))
            imgs.extend([yes, no])
            
        # save images into a matrix for displaying
        true_matrix = np.vstack(np.array(imgs[::2]))
        false_matrix = np.vstack(np.array(imgs[1::2]))

        # Plot with vertical grid
        fig = plt.figure(figsize=(25., 25.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(1, 4),  # creates 2x2 grid of axes
                        axes_pad=0.2,  # pad between axes in inch.
                        )

        grid[0].imshow(true_matrix)
        grid[0].set_title("Positive", fontsize=15)
        grid[0].axis("off")
        grid[1].imshow(false_matrix)
        grid[1].set_title("Negative", fontsize=15)
        grid[1].axis("off")

        grid[2].imshow(true_matrix[:,:,0], cmap="hot")
        grid[2].set_title("Positive", fontsize=15)
        grid[2].axis("off")
        grid[3].imshow(false_matrix[:,:,0], cmap="hot")
        grid[3].set_title("Negative", fontsize=15)
        grid[3].axis("off")

        # annotations
        plt.suptitle("Brain Tumor Detection / LGG Segmentation Dataset", 
                    y=.95, fontsize=(self.scale/30), weight="bold")
        plt.figtext(0.36,0.05,"Original", va="center", ha="center", size=20)
        plt.figtext(0.64,0.05,"Hot colormap", va="center", ha="center", size=20)

        # save and display
        plt.savefig(file_path + "png/sampleData.png", bbox_inches='tight', pad_inches=0.2, transparent=False)
        plt.show()

    #=======================================================================        
    def random_positive_patient (self, sampleSize):
        '''random display positive cases'''
        
        # select random images
        sample_df = self.frame[self.frame["status"] == 1].sample(sampleSize).values
        
        # resize and fit the size of images
        imgs = []
        for i, data in enumerate(sample_df):
            #print(data)
            img = cv2.resize(cv2.imread(data[0]), (self.scale, self.scale))
            mask = cv2.resize(cv2.imread(data[1]), (self.scale, self.scale))
            imgs.extend([img, mask])

        # save images into a matrix for displaying
        imgs_array = np.hstack(np.array(imgs[::2]))
        masks_array = np.hstack(np.array(imgs[1::2]))

        # Plot with horizental grid
        fig = plt.figure(figsize=(25., 25.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(3, 1),  # creates 2x2 grid of axes
                        axes_pad=0.1,  # pad between axes in inch.
                        )

        grid[0].imshow(imgs_array)
        grid[0].set_title("Images", fontsize=15)
        grid[0].axis("off")
        grid[1].imshow(masks_array)
        grid[1].set_title("Masks", fontsize=15, y=0.9)
        grid[1].axis("off")
        grid[2].imshow(imgs_array)
        grid[2].imshow(masks_array, alpha=0.4)
        grid[2].set_title('Brain MRI with mask', fontsize=15)
        grid[2].axis('off')

        # save and display
        plt.savefig(file_path + "png/samplePositive.png", bbox_inches='tight', pad_inches=0.2, transparent=False)
        plt.show()

    #=======================================================================
    def plot_dice_history(model_name, train_dice_history, val_dice_history, num_epochs):
        '''plotting the result of training datasets'''
        
        x = np.arange(num_epochs)
        plt.figure(figsize=(10, 6))
        plt.plot(x, train_dice_history, label='Train DICE', lw=3, c="b")
        plt.plot(x, val_dice_history, label='Validation DICE', lw=3, c="r")

        plt.title(f"{model_name}", fontsize=20)
        plt.legend(fontsize=12)
        plt.xlabel("Epoch", fontsize=15)
        plt.ylabel("DICE", fontsize=15)
        
        plt.savefig(file_path + "png/dice_history.png", bbox_inches='tight', pad_inches=0.2, transparent=False)
        plt.show()
         
    #=======================================================================

    def plot_loss_history(model_name, train_loss_history, val_loss_history, num_epochs):
        '''plotting loss function result'''
        
        x = np.arange(num_epochs)
        plt.figure(figsize=(10, 6))
        plt.plot(x, train_loss_history, label='Train Loss', lw=3, c="b")
        plt.plot(x, val_loss_history, label='Validation Loss', lw=3, c="r")

        plt.title(f"{model_name}", fontsize=20)
        plt.legend(fontsize=12)
        plt.xlabel("Epoch", fontsize=15)
        plt.ylabel("Loss", fontsize=15)
        
        plt.savefig(file_path + "png/loss_history.png", bbox_inches='tight', pad_inches=0.2, transparent=False)
        plt.show()
    #=======================================================================
    def plot_test_prediction (model, device, test, sample):
        '''plotting a sample of prediction'''
        
        test_sample = test[test["status"] == 1].values[sample]

        image = cv2.resize(cv2.imread(test_sample[0]), (128, 128))
        mask = cv2.resize(cv2.imread(test_sample[1]), (128, 128))

        # prediction
        pred = torch.tensor(image.astype(np.float32) / 255.).unsqueeze(0).permute(0,3,1,2)
        pred = tt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(pred)
        pred = model(pred.to(device)) 
        pred = pred.detach().cpu().numpy()[0,0,:,:] # convert tensor to the numpy
        
        # tresholding
        pred_t = np.copy(pred)
        pred_t[np.nonzero(pred_t < 0.3)] = 0.0
        pred_t[np.nonzero(pred_t >= 0.3)] = 255.
        pred_t = pred_t.astype("uint8")

        # plot
        fig , ax = plt.subplots(nrows=2,  ncols=2, figsize=(10, 10))

        ax[0, 0].imshow(image)
        ax[0, 0].set_title("image")
        ax[0, 1].imshow(mask)
        ax[0, 1].set_title("mask")
        ax[1, 0].imshow(pred)
        ax[1, 0].set_title("prediction")
        ax[1, 1].imshow(pred_t)
        ax[1, 1].set_title("prediction with threshold")
        
        plt.savefig(file_path + "png/random_pridict.png", bbox_inches='tight', pad_inches=0.2, transparent=False)
        plt.show()
    
