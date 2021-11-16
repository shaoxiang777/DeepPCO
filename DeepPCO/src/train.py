import os
import cv2
import torch
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.model import DeepPCO
from models.loss import MaskedMSELoss, MSELoss
from datasets.euroc import EuRoC
from datasets.kitti import KITTI
from torch.utils.tensorboard import SummaryWriter
from configs.cfg import args
from utils.utils import mkdir


def main(args=args, resume_train=False, save_epochwise=True, save_net=True):
    PATH = mkdir(args["save_path"])
    # # create a log writer
    writer = SummaryWriter(PATH)

    torch.set_printoptions(precision=10)
    # set device cuda or cpu
    torch.cuda.set_device(6)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device('cpu')

    print('Available devices ', torch.cuda.device_count())
    print('Current cuda device ', torch.cuda.current_device())

    # set network
    net = DeepPCO()
    net.to(device)

    # set optimizer
    optimizer = optim.Adam(net.parameters(),
                           lr=args['lr'],
                           betas=(0.9,0.999))

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 10, args["weight_decay"])

    if resume_train:
        net_path = args["net_path"]
        checkpoint = torch.load(net_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    net.to(device).train()

    # set loss function
    # criterion = MaskedMSELoss(weight=args["loss_weight"])
    criterion = MSELoss(weight=args["loss_weight"])

    train_set = KITTI("train")
    test_set = KITTI("test")

    torch.manual_seed(13)
    print(len(train_set))
    print(len(test_set))

    # train_set, test_set = torch.utils.data.random_split(data_set,
    #                                                     [train_data_len,
    #                                                      test_data_len + 1])

    train_loader = DataLoader(train_set,
                              batch_size=args['train_batch_size'],
                              num_workers=args['num_workers'],
                              shuffle=args['shuffle'])

    test_loader = DataLoader(test_set,
                             batch_size=args['train_batch_size'],
                             num_workers=args['num_workers'],
                             shuffle=False)

    # train loop
    for key, value in args.items():
        writer.add_text(str(key), str(value))

    writer_index_train = 0

    for epoch in range(args['epoch']):
        # if (epoch + 1) % 10 == 0 and epoch != 0:
        #     weight_decay = pow(args['weight_decay'], (epoch+1)/10)
        #     writer.add_scalar("Weight Decay", weight_decay, epoch)
        #     optimizer = optim.Adam(net.parameters(),
        #                 lr=args['lr'] * weight_decay,
        #                 betas=(0.9,0.999))
        train(net, epoch, train_loader, criterion, optimizer,
              writer, writer_index_train, device)
        test(net, epoch, test_loader, criterion, writer, device)
        lr_scheduler.step()
        writer_index_train += 1

        # save the network by each epoch

        if save_epochwise and save_net:
            net_path = os.path.join(
                PATH, 'net_parameters_epoch_' + str(epoch) + '.pth')
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, net_path)

    # save the net at end of the train
    if save_net:
        net_path = os.path.join(PATH, 'net_parameters_final' + '.pth')
        torch.save(net.state_dict(), net_path)

    writer.close()


def train(net, epoch, train_loader, criterion,
          optimizer, writer, writer_index_train, device):
    train_epoch_loss = 0.0

    len_train_loader = len(train_loader)
    with tqdm(total=len_train_loader) as epoch_bar_train:
        for i, data in enumerate(iter(train_loader)):
            optimizer.zero_grad()

            input = data[0].to(device)

            target_t = data[1].to(device)
            target_r = data[2].to(device)
            # # 0 for machine hall (no rotation gt), 1 for vicon room
            # dataset_idx = data[3].to(device)

            pred_t_t, pred_t_r, pred_r_t, pred_r_r = net(input)
            # print(target_t)
            # print(target_r)
            # print(pred_r_t.squeeze())
            # print(pred_r_r.squeeze())

            # loss_t: loss from t-subnet
            # loss_r: loss from r-subnet
            # loss_t = criterion(pred_t_t, target_t,
            #                    pred_t_r, target_r, dataset_idx)
            # loss_r = criterion(pred_r_t, target_t,
            #                    pred_r_r, target_r, dataset_idx)
            loss_t, t,b = criterion(pred_t_t, target_t,
                                    pred_t_r, target_r)
            loss_r, z, x = criterion(pred_r_t, target_t,
                                     pred_r_r, target_r)

            loss_train = loss_t + loss_r
            # if loss_train.item() > 10:
            #     print(target_t)
            #     print(target_r)
            #     print(pred_t_t)
            #     print(pred_r_t)
            #     print(pred_t_r)
            #     print(pred_r_r)

            train_epoch_loss += loss_train.item()

            # demonstrate the epoch bar
            epoch_bar_train.set_description(
                'Train Epoch: {epoch + 1} | Loss: {loss_train.item()}'
            )
            epoch_bar_train.update(1)

            # calculate gradient of loss
            loss_train.backward()
            optimizer.step()
            # update loearning rate
            # write logger
            # writer.add_histogram('weight1',
            #                      net.t_net.conv[1].conv_block[0].weight,
            #                      writer_index_train)
            # writer.add_histogram('weight2',
            #                      net.t_net.fc_t[7].fc_block[0].weight,
            #                      writer_index_train)
            # writer.add_histogram('weight3',
            #                      net.r_net.fc_r[7].fc_block[0].weight,
            #                      writer_index_train)
            # writer.add_histogram('weight4',
            #                      net.r_net.flownet_conv.conv_block1a[2][0].weight,
            #                      writer_index_train)
            # writer.add_histogram('weight5',
            #                      net.r_net.fc_t[7].fc_block[0].weight,
            #                      writer_index_train)
            # writer.add_histogram('weight6',
            #                      net.r_net.fc_r[7].fc_block[0].weight,
            #                      writer_index_train)

        writer.add_scalar('Train Loss', train_epoch_loss / len_train_loader,
                          epoch)

        # writer.add_graph(net, input)


def test(net, epoch, test_loader, criterion,
          writer, device):
    test_epoch_loss = 0.0
    test_epoch_r_loss = 0.0
    test_epoch_t_loss = 0.0
    test_epoch_best_loss = 0.0
    test_epoch_t = 0.0
    test_epoch_r = 0.0

    len_test_loader = len(test_loader)
    with tqdm(total=len_test_loader) as epoch_bar_test:
        for i, data in enumerate(iter(test_loader)):
            with torch.no_grad():
                input = data[0].to(device)

                target_t = data[1].to(device)
                target_r = data[2].to(device)

                # # 0 for machine hall (no rotation gt), 1 for vicon room
                # dataset_idx = data[3].to(device)

                pred_t_t, pred_t_r, pred_r_t, pred_r_r = net(input)

                # loss_t: loss from t-subnet
                # loss_r: loss from r-subnet
                # loss_t = criterion(pred_t_t, target_t,
                #                    pred_t_r, target_r, dataset_idx)
                # loss_r = criterion(pred_r_t, target_t,
                #                    pred_r_r, target_r, dataset_idx)
                # loss_best = criterion(pred_t_t, target_t,
                #                       pred_r_r, target_r, dataset_idx)
                loss_t, t,b = criterion(pred_t_t, target_t,
                                   pred_t_r, target_r)
                loss_r, z, x = criterion(pred_r_t, target_t,
                                   pred_r_r, target_r)
                loss_best, y, u = criterion(pred_t_t, target_t,
                                      pred_r_r, target_r)

                loss_test = loss_t + loss_r

                test_epoch_loss += loss_test.item()
                test_epoch_t_loss += loss_t.item()
                test_epoch_r_loss += loss_r.item()
                test_epoch_best_loss += loss_best.item()
                test_epoch_t += t.item()
                test_epoch_r += x.item()


                # demonstrate the epoch bar
                epoch_bar_test.set_description(
                    f'Test Epoch: {epoch + 1} | Loss: {loss_test.item()}')
                epoch_bar_test.update(1)

        writer.add_scalar('Test Loss', test_epoch_loss / len_test_loader,
                          epoch)
        writer.add_scalar('T Loss', test_epoch_t_loss / len_test_loader,
                          epoch)
        writer.add_scalar('R Loss', test_epoch_r_loss / len_test_loader,
                                                          epoch)
        writer.add_scalar('Best Loss', test_epoch_best_loss / len_test_loader,
                          epoch)

        writer.add_scalar('t', test_epoch_t / len_test_loader,
                          epoch)
        writer.add_scalar('r', test_epoch_r / len_test_loader,
                          epoch)


if __name__ == '__main__':
    main()
