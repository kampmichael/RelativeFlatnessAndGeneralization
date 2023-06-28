from monai.losses import DiceLoss
import numpy as np
import argparse
from prettytable import PrettyTable
from LinearAE import LinearLatentAE
from data_utils_skullrec import *

#######
import torch
from FAMloss import FAMreg, LayerHessian
###########

'''
# training
1) download the dataset at https://uni-duisburg-essen.sciebo.de/s/gPUJycbAIvwog7z/download, and unzip
2) command line: python run_training_SkullRec.py --phase train

# test: generate prediction files on the test set
command line: python run_training_SkullRec.py --phase test

# evaluation: print prediction accuracy for each test case
command line: python run_training_SkullRec.py --phase evaluation

toy dataset download link https://uni-duisburg-essen.sciebo.de/s/8V3LYNPctTEBMCB
'''


def dc(input1, input2):

    input1 = np.atleast_1d(input1.astype(np.bool))
    input2 = np.atleast_1d(input2.astype(np.bool))
    
    intersection = np.count_nonzero(input1 & input2)
    
    size_i1 = np.count_nonzero(input1)
    size_i2 = np.count_nonzero(input2)
    
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0
    
    return dc



class train_utils(object):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0")
        self.model = LinearLatentAE(
            dimensions=3,
            in_shape=(1,256,256,128),
            out_channels=2,
            channels=(32, 64, 128, 64, 32, 8),
            latent_size=32,
            #latent_size=2,
            strides=(2, 2, 2, 2, 2, 2),
            num_res_units=0,
            #norm=Norm.BATCH,
        ).to(self.device) 
        self.optimizer = torch.optim.Adam(self.model.parameters(), 1e-4)

    def count_parameters(self,model):
        table = PrettyTable(['Modules', 'Parameters'])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        #print(table)
        print(f'Total Trainable Params: {total_params}')
        return total_params



    def loss_fam(self, input_, outputs, ys, feature_layer_id, network):
        is_training=1
        lmb = 0.1
        #ys=torch.squeeze(ys, 1)
        criterion = DiceLoss(to_onehot_y=True, softmax=True, reduction='sum')
        #criterion = F.cross_entropy
        if is_training: 
            layer_hessian = LayerHessian(network, feature_layer_id, criterion, method="functorch")
            loss = 80*criterion(outputs, ys) + lmb * FAMreg(input_, outputs, ys, layer_hessian,
                                                         norm_function='layerwise_trace', approximate=True)
        else:
            loss = 80*criterion(outputs, ys)
        return loss


    def loss_baseline(self,outputs, ys):
        criterion = DiceLoss(to_onehot_y=True, softmax=True,reduction='sum').to(device)
        return 80*criterion(outputs, ys)


    def test_featurelayerID(self):

        for i, p in enumerate(self.model.named_parameters()):
            print('i',i)
            print('p0',p[0])
            print('p1',p[1].shape)
            print('################################')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase")
    args = parser.parse_args()
    trainUtil=train_utils()

    model=trainUtil.model
    optimizer=trainUtil.optimizer
    loss_fcn=trainUtil.loss_fam
    #loss_fcn=trainUtil.loss_baseline
    device=trainUtil.device

    weights_dir='./ckpt/'
    
    if True: # args.phase=='train':
        max_epochs = 200
        epoch_loss_values = [] 
        print('start traininig...')

        # uncomment to continue training from a previous ckeckpoint
        #model.load_state_dict(torch.load(os.path.join('./ckpt/', "ckpt.pth"),map_location='cuda:0'))

        feature_layer_id=37


        for epoch in range(max_epochs):
            print(f"#######################epoch {epoch + 1}/{max_epochs}")
            #trainUtil.count_parameters(model)
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs, labels = (  
                    batch_data["image"].to(device),   
                    batch_data["label"].to(device),
                )
                optimizer.zero_grad()
                recon_batch = model(inputs)
                
                # use baseline or FAM regularized loss
                loss=loss_fcn(inputs, recon_batch, labels, feature_layer_id, model)
                #loss=loss_fcn(recon_batch, labels)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                print(
                    f"{step}/{len(train_ds) // train_loader.batch_size}, "
                    f"overall train_loss: {loss.item():.4f}")

            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"**************************epoch {epoch + 1} average loss: {epoch_loss:.4f}") 

        torch.save(model.state_dict(), os.path.join(weights_dir, "skullrec_ckpt.pth"))
        epoch_loss_values=np.array(epoch_loss_values)
        np.save('epoch_loss_values.npy',epoch_loss_values)

  


    elif args.phase=='test':
        print('###############generating predictions on the test set')

        model.load_state_dict(torch.load(os.path.join(weights_dir, "skullrec_ckpt.pth"),map_location='cuda:0'))
        model.eval()
        with torch.no_grad():
            for test_file in test_org_loader:
                test_inputs = test_file["image"].to(device)
                inferer=SimpleInferer()
                test_file["pred"]= inferer(test_inputs, model)
                test_file = [post_transforms(i) for i in decollate_batch(test_file)]
                test_output = from_engine(["pred"])(test_file)



    elif args.phase=='evaluation':
        print('#############printing test accuracy')
        #  prediction file path './output'
        pathlist = Path('./output').glob('**/*.nii.gz')
        if not os.path.exists('./predictions/'):
            os.mkdir('./predictions/')

        i=0
        for path in pathlist:
             path_in_str = str(path)
             shutil.copyfile(path_in_str, './predictions/'+'{0:03}'.format(i)+'.nii.gz')
             i+=1


        predictions = sorted(glob.glob(os.path.join('./predictions/', "*.nii.gz")))



        for j in range(len(predictions)):
            gt=test_label[j]
            pred=predictions[j]

            print('gt',gt)
            print('pred',pred)

            predNib = nib.load(pred)
            predImg = predNib.get_data()
            gtNib= nib.load(gt)
            gtImg = gtNib.get_data()

            print(dc(predImg,gtImg))