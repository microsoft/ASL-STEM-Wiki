from datasets import *
from models import *
from accelerate import Accelerator
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import random
from collections import namedtuple


wandb.login(key='YOUR_KEY_HERE', relogin=True)

def compute_iou(predictions, labels):
    iou_score = 0
    for prediction, label in zip(predictions, labels):
        assert len(list(prediction)) == len(list(label)), "Sequences must be of equal length"

        # get first index of 2 in label
        try:
            start = label.index(2)
        except:
            start = len(label)
        label = label[:start]
        prediction = prediction[:start]
        
        intersection = sum(p == 1 and l == 1 for p, l in zip(prediction, label))
        union = sum(p == 1 or l == 1 for p, l in zip(prediction, label))
        
        iou_score += intersection / union if union != 0 else 0
    return iou_score / len(predictions)

def singletask(cfg):
    torch.manual_seed(cfg.seed)
    device = cfg.meta.device
    accelerator = Accelerator()

    checkpoint_path = Path(f'{cfg.train.model_dir}/checkpoints')
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)
    checkpoint_path = Path(f'{cfg.finetune.model_dir}/checkpoints')
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)
    output_path = Path(f'{cfg.finetune.model_dir}/outputs')
    if not output_path.exists():
        output_path.mkdir(parents=True)

    ds = ASLWikiDataset(datadir=cfg.data.datadir, label_file=cfg.data.label_file, min_frames=cfg.data.clip_length * 2 + 100)

    train_ft_ds = ASLWikiDataset(datadir=cfg.data.ft_datadir, label_file=cfg.data.ft_label_file, min_frames=0, eval_article=cfg.data.eval_article, valid=False)
    val_ft_ds = ASLWikiDataset(datadir=cfg.data.ft_datadir, label_file=cfg.data.ft_label_file, min_frames=0, eval_article=cfg.data.eval_article, valid=True)

    train_ft_onespan_ds = ASLWikiDataset(datadir=cfg.data.ft_datadir, label_file=cfg.data.ft_label_file.replace('train', 'train-onespan'), 
                                         min_frames=0, eval_article=cfg.data.eval_article, valid=False)
    val_ft_onespan_ds = ASLWikiDataset(datadir=cfg.data.ft_datadir, label_file=cfg.data.ft_label_file.replace('train', 'train-onespan'),
                                        min_frames=0, eval_article=cfg.data.eval_article, valid=True)

    train_size = int(0.9 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size]) # not randomized - ok?
    train_ds = train_ds.dataset
    val_ds = val_ds.dataset

    contra_video_dataset = ContraVideoDataset(train_ds, clip_length=cfg.data.clip_length, seq_length=2*cfg.data.seq_length)
    contra_text_dataset = ContraTextDataset(train_ds, clip_length=2*cfg.data.clip_length, seq_length=cfg.data.seq_length)
    ft_video_dataset = SpanVideoDataset(train_ft_ds, clip_length=2*cfg.data.clip_length, seq_length=cfg.data.seq_length)
    ft_text_dataset = SpanTextDataset(train_ft_onespan_ds, clip_length=2*cfg.data.clip_length, seq_length=cfg.data.seq_length)

    valid_contra_video_dataset = ContraVideoDataset(val_ds, clip_length=cfg.data.clip_length, seq_length=2*cfg.data.seq_length)
    valid_contra_text_dataset = ContraTextDataset(val_ds, clip_length=2*cfg.data.clip_length, seq_length=cfg.data.seq_length)
    valid_ft_video_dataset = SpanVideoDataset(val_ft_ds, clip_length=2*cfg.data.clip_length, seq_length=cfg.data.seq_length)
    valid_ft_text_dataset = SpanTextDataset(val_ft_onespan_ds, clip_length=2*cfg.data.clip_length, seq_length=cfg.data.seq_length)

    contra_video_dataloader = DataLoader(contra_video_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    contra_text_dataloader = DataLoader(contra_text_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    ft_video_dataloader = DataLoader(ft_video_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    ft_text_dataloader = DataLoader(ft_text_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=1, pin_memory=True)

    valid_contra_video_dataloader = DataLoader(valid_contra_video_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    valid_contra_text_dataloader = DataLoader(valid_contra_text_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    valid_ft_video_dataloader = DataLoader(valid_ft_video_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    valid_ft_text_dataloader = DataLoader(valid_ft_text_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=1, pin_memory=True)

    print(f'Number of samples: {len(contra_video_dataset)} / {len(contra_text_dataset)} / {len(ft_video_dataset)} / {len(ft_text_dataset)}')

    model = MultiTaskModel(input_feature=2*cfg.model.num_keypoints, hidden_feature=cfg.model.hidden_feature, clip_length=2*cfg.data.clip_length, 
            seq_length=cfg.data.seq_length, p_dropout=cfg.model.p_dropout, num_stage=cfg.model.num_stages).to(device)
    
    criterion = nn.CrossEntropyLoss()
    binary_criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    model, optimizer = accelerator.prepare(model, optimizer)
    contra_video_dataloader = accelerator.prepare(contra_video_dataloader)
    contra_text_dataloader = accelerator.prepare(contra_text_dataloader)
    ft_video_dataloader = accelerator.prepare(ft_video_dataloader)
    ft_text_dataloader = accelerator.prepare(ft_text_dataloader)

    name = cfg.wandb.name
    wandb.init(project='asl-stem-wiki',
               entity='YOUR_ENTITY',
               name=name)

    ## edge case for 0 epochs
    text_loss = []
    text_acc = []

    video_loss = []
    video_acc = []

    for epoch in range(cfg.train.num_epochs):
        text_loss = []
        text_acc = []

        video_loss = []
        video_acc = []

        model.train()
        for batch_idx, (frames, text, attention_mask, label) in enumerate(contra_text_dataloader):
            optimizer.zero_grad()

            task = 'text_clf'
            frames = frames[:,:,:,:2]
            b, s, k, d = frames.shape
            if b == 1:
                continue
            frames = frames.reshape(b, s, -1)

            # Forward pass
            outputs = model(frames.to(device), text.to(device), attention_mask.to(device), task)

            # Calculate the loss
            loss = criterion(outputs, label.to(device))

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            text_loss.append(loss.item())

            preds = torch.argmax(outputs, dim=1)
            acc = np.mean(preds.detach().cpu().numpy() == label.detach().cpu().numpy())
            text_acc.append(acc)

        loss = sum(text_loss) / max(1,len(text_loss))
        acc = sum(text_acc) / max(1,len(text_acc))
        print(f'Epoch [{epoch+1}/{cfg.train.num_epochs}], Batch [{batch_idx}/{len(contra_text_dataloader)}], Loss: {loss}, Acc: {acc}')
        wandb.log({'epoch': (epoch + 1) , 'batch': (batch_idx) / len(contra_text_dataloader), 'train_loss_text': loss, 'train_acc_text': acc})
        

        for batch_idx, (clip1, clip2, text, attention_mask, label) in enumerate(contra_video_dataloader):
            optimizer.zero_grad()

            task = 'video_clf'
            clip1 = clip1[:,:,:,:2]
            clip2 = clip2[:,:,:,:2]
            b, s, k, d = clip1.shape
            if b == 1:
                continue
            frames = torch.cat([clip1.reshape(b, s, -1), clip2.reshape(b, s, -1)], dim=1)

            # Forward pass
            outputs = model(frames.to(device), text.to(device), attention_mask.to(device), task)

            # Calculate the loss
            loss = criterion(outputs, label.to(device))

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            video_loss.append(loss.item())

            preds = torch.argmax(outputs, dim=1)
            acc = np.mean(preds.detach().cpu().numpy() == label.detach().cpu().numpy())
            video_acc.append(acc)
        
        loss = sum(video_loss) / max(1, len(video_loss))
        acc = sum(video_acc) / max(1, len(video_acc))
        # Print the loss
        preds = torch.argmax(outputs, dim=1)
        acc = np.mean(preds.detach().cpu().numpy() == label.detach().cpu().numpy())
        print(f'Epoch [{epoch+1}/{cfg.train.num_epochs}], Batch [{batch_idx}/{len(contra_video_dataloader)}], Loss: {loss}, Acc: {acc}')
        wandb.log({'epoch': (epoch + 1) , 'batch': (batch_idx) / len(contra_video_dataloader), 'train_loss_video': loss, 'train_acc_video': acc})


        # Compute validation loss
        model.eval()
        text_loss = []
        text_acc = []

        video_loss = []
        video_acc = []

        for batch_idx, (frames, text, attention_mask, label) in enumerate(valid_contra_text_dataloader):
            task = 'text_clf'
            frames = frames[:,:,:,:2]
            b, s, k, d = frames.shape
            if b == 1:
                continue
            frames = frames.reshape(b, s, -1)

            # Forward pass
            outputs = model(frames.to(device), text.to(device), attention_mask.to(device), task)

            # Calculate the loss
            loss = criterion(outputs, label.to(device))
            text_loss.append(loss.item())

            preds = torch.argmax(outputs, dim=1)
            acc = np.mean(preds.detach().cpu().numpy() == label.detach().cpu().numpy())
            text_acc.append(acc)

        loss = sum(text_loss) / max(1,len(text_loss))
        acc = sum(text_acc) / max(1,len(text_acc))
        print(f'Epoch [{epoch+1}/{cfg.train.num_epochs}], Batch [{batch_idx}/{len(valid_contra_text_dataloader)}], Loss: {loss}, Acc: {acc}')
        wandb.log({'epoch': (epoch + 1) , 'batch': (batch_idx) / len(valid_contra_text_dataloader), 'valid_loss_text': loss, 'valid_acc_text': acc})

        for batch_idx, (clip1, clip2, text, attention_mask, label) in enumerate(valid_contra_video_dataloader):
            task = 'video_clf'
            clip1 = clip1[:,:,:,:2]
            clip2 = clip2[:,:,:,:2]
            b, s, k, d = clip1.shape
            if b == 1:
                continue
            frames = torch.cat([clip1.reshape(b, s, -1), clip2.reshape(b, s, -1)], dim=1)

            # Forward pass
            outputs = model(frames.to(device), text.to(device), attention_mask.to(device), task)

            # Calculate the loss
            loss = criterion(outputs, label.to(device))
            video_loss.append(loss.item())

            preds = torch.argmax(outputs, dim=1)
            acc = np.mean(preds.detach().cpu().numpy() == label.detach().cpu().numpy())
            video_acc.append(acc)

        loss = sum(video_loss) / max(1,len(video_loss))
        acc = sum(video_acc) / max(1,len(video_acc))
        print(f'Epoch [{epoch+1}/{cfg.train.num_epochs}], Batch [{batch_idx}/{len(valid_contra_video_dataloader)}], Loss: {loss}, Acc: {acc}')
        wandb.log({'epoch': (epoch + 1) , 'batch': (batch_idx) / len(valid_contra_video_dataloader), 'valid_loss_video': loss, 'valid_acc_video': acc})


        # Save the model checkpoint
        try:
            torch.save({'state_dict': model.state_dict(),
                        'hyperparameters': cfg,
                        }, f'{cfg.train.model_dir}/checkpoints/{cfg.data.eval_article}-model.pth')
        except:
            pass
        wandb.save(f'{cfg.train.model_dir}/checkpoints/{cfg.data.eval_article}-model.pth')

    mean_text_loss = sum(text_loss) / max(1, len(text_loss))
    mean_video_loss = sum(video_loss) / max(1, len(video_loss))
    print(f'Text Loss: {mean_text_loss}, Video Loss: {mean_video_loss}')

    detect_loss = []
    detect_acc = []
    align_loss = []
    align_acc = []

    # reset optimizer
    detect_optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    detect_model, optimizer = accelerator.prepare(model, optimizer)

    # copy model
    align_model = MultiTaskModel(input_feature=2*cfg.model.num_keypoints, hidden_feature=cfg.model.hidden_feature, clip_length=2*cfg.data.clip_length, 
            seq_length=cfg.data.seq_length, p_dropout=cfg.model.p_dropout, num_stage=cfg.model.num_stages).to(device)
    
    align_model.load_state_dict(model.state_dict())
    align_optimizer = optim.Adam(align_model.parameters(), lr=cfg.train.learning_rate)
    align_model, optimizer = accelerator.prepare(align_model, align_optimizer)


    for epoch in range(cfg.finetune.num_epochs):
        detect_loss = []
        detect_acc = []
        align_loss = []
        align_acc = []
        model.train()

        for batch_idx, (frames, text, attention_mask, label, start_positions, end_positions) in enumerate(ft_text_dataloader):
            align_optimizer.zero_grad()

            task = 'text_span'
            frames = frames[:,:,:,:2]
            b, s, k, d = frames.shape
            if b == 1:
                continue
            frames = frames.reshape(b, s, -1)

            # Forward pass
            outputs = align_model(frames.to(device), text.to(device), attention_mask.to(device), task)#.permute(0, 2, 1)
            start_logits, end_logits = outputs.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)  # [batch_size, sequence_length]
            end_logits = end_logits.squeeze(-1)      # [batch_size, sequence_length]
            
            # Calculate the loss
            # print("Output and label shapes:", outputs.shape, label.shape)
            # token_loss = binary_criterion(outputs, label.to(device))
            # one_mask = label == 1
            # zero_mask = label == 0
            # one_masked_loss = token_loss * one_mask.to(device)
            # zero_masked_loss = token_loss * zero_mask.to(device)

            # loss = one_masked_loss.mean() + zero_masked_loss.mean()

            start_loss = criterion(start_logits, start_positions.to(device))
            end_loss = criterion(end_logits, end_positions.to(device))
            loss = (start_loss + end_loss) / 2

            # Backpropagation and optimization
            loss.backward()
            align_optimizer.step()
            align_loss.append(loss.item())

            start_preds = torch.argmax(start_logits, dim=1)
            end_preds = torch.argmax(end_logits, dim=1)
            preds = torch.zeros_like(label)
            for i in range(len(start_preds)):
                preds[i, start_preds[i]:end_preds[i]+1] = 1
            acc = compute_iou(preds.detach().cpu().numpy(), label.detach().cpu().numpy())
            align_acc.append(acc)


        # Print the loss
        loss = sum(align_loss) / len(align_loss)
        acc = sum(align_acc) / len(align_acc)

        print(f'Epoch [{epoch+1}/{cfg.finetune.num_epochs}], Batch [{batch_idx}/{len(ft_text_dataloader)}], Loss: {loss}, Acc: {acc}')
        wandb.log({'epoch': (epoch + 1), 'batch': (batch_idx) / len(ft_text_dataloader), 'train_loss_align': loss, 'train_acc_align': acc})

        for batch_idx, (frames, text, attention_mask, label) in enumerate(ft_video_dataloader):
            detect_optimizer.zero_grad()

            task = 'video_span'
            frames = frames[:,:,:,:2]
            b, s, k, d = frames.shape
            if b == 1:
                continue
            frames = frames.reshape(b, s, -1)

            # Forward pass
            outputs = detect_model(frames.to(device), text.to(device), attention_mask.to(device), task).permute(0, 2, 1)

            # Calculate the loss
            token_loss = binary_criterion(outputs, label.to(device))
            one_mask = label == 1
            zero_mask = label == 0
            one_masked_loss = token_loss * one_mask.to(device)
            zero_masked_loss = token_loss * zero_mask.to(device)
            loss = one_masked_loss.mean() + zero_masked_loss.mean()

            # Backpropagation and optimization
            loss.backward()
            detect_optimizer.step()
            detect_loss.append(loss.item())

            preds = torch.argmax(outputs, dim=1)
            acc = compute_iou(preds.detach().cpu().numpy(), label.detach().cpu().numpy())
            detect_acc.append(acc)

        # Print the loss
        loss = sum(detect_loss) / len(detect_loss)
        acc = sum(detect_acc) / len(detect_acc)
        print(f'Epoch [{epoch+1}/{cfg.finetune.num_epochs}], Batch [{batch_idx}/{len(ft_video_dataloader)}], Loss: {loss}, Acc: {acc}')
        wandb.log({'epoch': (epoch + 1), 'batch': (batch_idx) / len(ft_video_dataloader), 'train_loss_detect': loss, 'train_acc_detect': acc})


        # Compute validation
        detect_model.eval()
        align_model.eval()
        detect_loss = []
        detect_acc = []
        detect_preds = []
        detect_labels = []
        align_loss = []
        align_acc = []
        align_preds = []
        align_labels = []

        for batch_idx, (frames, text, attention_mask, label, start_positions, end_positions) in enumerate(valid_ft_text_dataloader):
            align_optimizer.zero_grad()

            task = 'text_span'
            frames = frames[:,:,:,:2]
            b, s, k, d = frames.shape
            if b == 1:
                continue
            frames = frames.reshape(b, s, -1)

            # Forward pass
            outputs = align_model(frames.to(device), text.to(device), attention_mask.to(device), task)#.permute(0, 2, 1)
            start_logits, end_logits = outputs.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)  # [batch_size, sequence_length]
            end_logits = end_logits.squeeze(-1)      # [batch_size, sequence_length]
            
            # Calculate the loss
            # print("Output and label shapes:", outputs.shape, label.shape)
            # token_loss = binary_criterion(outputs, label.to(device))
            # one_mask = label == 1
            # zero_mask = label == 0
            # one_masked_loss = token_loss * one_mask.to(device)
            # zero_masked_loss = token_loss * zero_mask.to(device)

            # loss = one_masked_loss.mean() + zero_masked_loss.mean()

            start_loss = criterion(start_logits, start_positions.to(device))
            end_loss = criterion(end_logits, end_positions.to(device))
            loss = (start_loss + end_loss) / 2
            
            align_loss.append(loss.item())

            start_preds = torch.argmax(start_logits, dim=1)
            end_preds = torch.argmax(end_logits, dim=1)
            preds = torch.zeros_like(label)
            for i in range(len(start_preds)):
                preds[i, start_preds[i]:end_preds[i]+1] = 1
            acc = compute_iou(preds.detach().cpu().numpy(), label.detach().cpu().numpy())
            align_acc.append(acc)

            for pred in preds:
                align_preds.append(list(map(lambda x: str(x.item()), pred)))
            for l in label:
                align_labels.append(list(map(lambda x: str(x.item()), l)))

        # Print the loss
        loss = sum(align_loss) / max(1, len(align_loss))
        acc = sum(align_acc) / max(1, len(align_acc))
        print(f'Epoch [{epoch+1}/{cfg.finetune.num_epochs}], Batch [{batch_idx}/{len(valid_ft_text_dataloader)}], Loss: {loss}, Acc: {acc}')
        wandb.log({'epoch': (epoch + 1), 'batch': (batch_idx) / len(valid_ft_text_dataloader), 'valid_loss_align': loss, 'valid_acc_align': acc})
        

        for batch_idx, (frames, text, attention_mask, label) in enumerate(valid_ft_video_dataloader):
            task = 'video_span'
            frames = frames[:,:,:,:2]
            b, s, k, d = frames.shape
            if b == 1:
                continue
            frames = frames.reshape(b, s, -1)

            # Forward pass
            outputs = detect_model(frames.to(device), text.to(device), attention_mask.to(device), task).permute(0, 2, 1)

            # Calculate the loss
            token_loss = binary_criterion(outputs, label.to(device))
            one_mask = label == 1
            zero_mask = label == 0
            one_masked_loss = token_loss * one_mask.to(device)
            zero_masked_loss = token_loss * zero_mask.to(device)
            loss = one_masked_loss.mean() + zero_masked_loss.mean()
            
            detect_loss.append(loss.item())

            preds = torch.argmax(outputs, dim=1)
            acc = compute_iou(preds.detach().cpu().numpy(), label.detach().cpu().numpy())
            detect_acc.append(acc)

            for pred in preds:
                detect_preds.append(list(map(lambda x: str(x.item()), pred)))
            for l in label:
                detect_labels.append(list(map(lambda x: str(x.item()), l)))

        # Print the loss
        loss = sum(detect_loss) / len(detect_loss)
        acc = sum(detect_acc) / len(detect_acc)
        print(f'Epoch [{epoch+1}/{cfg.finetune.num_epochs}], Batch [{batch_idx}/{len(valid_ft_video_dataloader)}], Loss: {loss}, Acc: {acc}')
        wandb.log({'epoch': (epoch + 1), 'batch': (batch_idx) / len(valid_ft_video_dataloader), 'valid_loss_detect': loss, 'valid_acc_detect': acc})


        # Save the model checkpoint
        try:
            torch.save({'state_dict': detect_model.state_dict(),
                        'hyperparameters': cfg,
                        }, f'{cfg.finetune.model_dir}/checkpoints/{cfg.data.eval_article}-detect-model.pth')
            torch.save({'state_dict': align_model.state_dict(),
                        'hyperparameters': cfg,
                        }, f'{cfg.finetune.model_dir}/checkpoints/{cfg.data.eval_article}-align-model.pth')
        except:
            pass
        with open(f'{cfg.finetune.model_dir}/outputs/{cfg.data.eval_article}-detect_preds.txt', 'w') as f:
            for pred in detect_preds:
                f.write(','.join(pred) + '\n')
        with open(f'{cfg.finetune.model_dir}/outputs/{cfg.data.eval_article}-detect_labels.txt', 'w') as f:
            for label in detect_labels:
                f.write(','.join(label) + '\n')
        with open(f'{cfg.finetune.model_dir}/outputs/{cfg.data.eval_article}-align_preds.txt', 'w') as f:
            for pred in align_preds:
                f.write(','.join(pred) + '\n')
        with open(f'{cfg.finetune.model_dir}/outputs/{cfg.data.eval_article}-align_labels.txt', 'w') as f:
            for label in align_labels:
                f.write(','.join(label) + '\n')

        wandb.save(f'{cfg.finetune.model_dir}/checkpoints/{cfg.data.eval_article}-detect-model.pth')
        wandb.save(f'{cfg.finetune.model_dir}/checkpoints/{cfg.data.eval_article}-align-model.pth')
        wandb.save(f'{cfg.finetune.model_dir}/outputs/{cfg.data.eval_article}-detect_preds.txt')
        wandb.save(f'{cfg.finetune.model_dir}/outputs/{cfg.data.eval_article}-detect_labels.txt')
        wandb.save(f'{cfg.finetune.model_dir}/outputs/{cfg.data.eval_article}-align_preds.txt')
        wandb.save(f'{cfg.finetune.model_dir}/outputs/{cfg.data.eval_article}-align_labels.txt')

    mean_detect_loss = sum(detect_loss) / len(detect_loss)
    mean_align_loss = sum(align_loss) / len(align_loss)
    print(f'Detect Loss: {mean_detect_loss}, Align Loss: {mean_align_loss}')
    wandb.finish()


    return mean_align_loss

def multitask(cfg):
    torch.manual_seed(cfg.seed)
    device = cfg.meta.device
    accelerator = Accelerator()

    checkpoint_path = Path(f'{cfg.train.model_dir}/checkpoints')
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)
    checkpoint_path = Path(f'{cfg.finetune.model_dir}/checkpoints')
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)
    output_path = Path(f'{cfg.finetune.model_dir}/outputs')
    if not output_path.exists():
        output_path.mkdir(parents=True)
    
    print("Loading datasets")

    ds = ASLWikiDataset(datadir=cfg.data.datadir, label_file=cfg.data.label_file, min_frames=cfg.data.clip_length * 2 + 100)

    train_ft_ds = ASLWikiDataset(datadir=cfg.data.ft_datadir, label_file=cfg.data.ft_label_file, min_frames=0, eval_article=cfg.data.eval_article, valid=False)
    val_ft_ds = ASLWikiDataset(datadir=cfg.data.ft_datadir, label_file=cfg.data.ft_label_file, min_frames=0, eval_article=cfg.data.eval_article, valid=True)

    train_ft_onespan_ds = ASLWikiDataset(datadir=cfg.data.ft_datadir, label_file=cfg.data.ft_label_file.replace('train', 'train-onespan'), 
                                         min_frames=0, eval_article=cfg.data.eval_article, valid=False)
    val_ft_onespan_ds = ASLWikiDataset(datadir=cfg.data.ft_datadir, label_file=cfg.data.ft_label_file.replace('train', 'train-onespan'),
                                        min_frames=0, eval_article=cfg.data.eval_article, valid=True)
    
    print("Splitting datasets")

    train_size = int(0.9 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size]) # not randomized - ok?
    train_ds = train_ds.dataset
    val_ds = val_ds.dataset

    contra_video_dataset = ContraVideoDataset(train_ds, clip_length=cfg.data.clip_length, seq_length=2*cfg.data.seq_length)
    contra_text_dataset = ContraTextDataset(train_ds, clip_length=2*cfg.data.clip_length, seq_length=cfg.data.seq_length)
    ft_video_dataset = SpanVideoDataset(train_ft_ds, clip_length=2*cfg.data.clip_length, seq_length=cfg.data.seq_length)
    ft_text_dataset = SpanTextDataset(train_ft_onespan_ds, clip_length=2*cfg.data.clip_length, seq_length=cfg.data.seq_length)

    valid_contra_video_dataset = ContraVideoDataset(val_ds, clip_length=cfg.data.clip_length, seq_length=2*cfg.data.seq_length)
    valid_contra_text_dataset = ContraTextDataset(val_ds, clip_length=2*cfg.data.clip_length, seq_length=cfg.data.seq_length)
    valid_ft_video_dataset = SpanVideoDataset(val_ft_ds, clip_length=2*cfg.data.clip_length, seq_length=cfg.data.seq_length)
    valid_ft_text_dataset = SpanTextDataset(val_ft_onespan_ds, clip_length=2*cfg.data.clip_length, seq_length=cfg.data.seq_length)

    contra_video_dataloader = DataLoader(contra_video_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    contra_text_dataloader = DataLoader(contra_text_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    ft_video_dataloader = DataLoader(ft_video_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    ft_text_dataloader = DataLoader(ft_text_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=1, pin_memory=True)

    valid_contra_video_dataloader = DataLoader(valid_contra_video_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    valid_contra_text_dataloader = DataLoader(valid_contra_text_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    valid_ft_video_dataloader = DataLoader(valid_ft_video_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    valid_ft_text_dataloader = DataLoader(valid_ft_text_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=1, pin_memory=True)

    print(f'Number of samples: {len(contra_video_dataset)} / {len(contra_text_dataset)} / {len(ft_video_dataset)} / {len(ft_text_dataset)}')

    model = MultiTaskModel(input_feature=2*cfg.model.num_keypoints, hidden_feature=cfg.model.hidden_feature, clip_length=2*cfg.data.clip_length, 
                         seq_length=cfg.data.seq_length, p_dropout=cfg.model.p_dropout, num_stage=cfg.model.num_stages).to(device)
    criterion = nn.CrossEntropyLoss()
    binary_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=2)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    model, optimizer = accelerator.prepare(model, optimizer)
    contra_video_dataloader = accelerator.prepare(contra_video_dataloader)
    contra_text_dataloader = accelerator.prepare(contra_text_dataloader)
    ft_video_dataloader = accelerator.prepare(ft_video_dataloader)
    ft_text_dataloader = accelerator.prepare(ft_text_dataloader)

    name = cfg.wandb.name
    wandb.init(project='aslnn',
               entity='kayo',
               name=name)

    ## edge case for 0 epochs
    text_loss = []
    text_acc = []

    video_loss = []
    video_acc = []

    for epoch in range(cfg.train.num_epochs):
        text_loss = []
        text_acc = []

        video_loss = []
        video_acc = []

        model.train()
        for batch_idx, (frames, text, attention_mask, label) in enumerate(contra_text_dataloader):
            optimizer.zero_grad()

            task = 'text_clf'
            frames = frames[:,:,:,:2]
            b, s, k, d = frames.shape
            if b == 1:
                continue
            frames = frames.reshape(b, s, -1)

            # Forward pass
            outputs = model(frames.to(device), text.to(device), attention_mask.to(device), task)

            # Calculate the loss
            loss = criterion(outputs, label.to(device))

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            text_loss.append(loss.item())

            preds = torch.argmax(outputs, dim=1)
            acc = np.mean(preds.detach().cpu().numpy() == label.detach().cpu().numpy())
            text_acc.append(acc)

        loss = sum(text_loss) / max(1,len(text_loss))
        acc = sum(text_acc) / max(1,len(text_acc))
        print(f'Epoch [{epoch+1}/{cfg.train.num_epochs}], Batch [{batch_idx}/{len(contra_text_dataloader)}], Loss: {loss}, Acc: {acc}')
        wandb.log({'epoch': (epoch + 1) , 'batch': (batch_idx) / len(contra_text_dataloader), 'train_loss_text': loss, 'train_acc_text': acc})
        

        for batch_idx, (clip1, clip2, text, attention_mask, label) in enumerate(contra_video_dataloader):
            optimizer.zero_grad()

            task = 'video_clf'
            clip1 = clip1[:,:,:,:2]
            clip2 = clip2[:,:,:,:2]
            b, s, k, d = clip1.shape
            if b == 1:
                continue
            frames = torch.cat([clip1.reshape(b, s, -1), clip2.reshape(b, s, -1)], dim=1)

            # Forward pass
            outputs = model(frames.to(device), text.to(device), attention_mask.to(device), task)

            # Calculate the loss
            loss = criterion(outputs, label.to(device))

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            video_loss.append(loss.item())

            preds = torch.argmax(outputs, dim=1)
            acc = np.mean(preds.detach().cpu().numpy() == label.detach().cpu().numpy())
            video_acc.append(acc)
        
        loss = sum(video_loss) / max(1, len(video_loss))
        acc = sum(video_acc) / max(1, len(video_acc))
        # Print the loss
        preds = torch.argmax(outputs, dim=1)
        acc = np.mean(preds.detach().cpu().numpy() == label.detach().cpu().numpy())
        print(f'Epoch [{epoch+1}/{cfg.train.num_epochs}], Batch [{batch_idx}/{len(contra_video_dataloader)}], Loss: {loss}, Acc: {acc}')
        wandb.log({'epoch': (epoch + 1) , 'batch': (batch_idx) / len(contra_video_dataloader), 'train_loss_video': loss, 'train_acc_video': acc})


        # Compute validation loss
        model.eval()
        text_loss = []
        text_acc = []

        video_loss = []
        video_acc = []

        for batch_idx, (frames, text, attention_mask, label) in enumerate(valid_contra_text_dataloader):
            task = 'text_clf'
            frames = frames[:,:,:,:2]
            b, s, k, d = frames.shape
            if b == 1:
                continue
            frames = frames.reshape(b, s, -1)

            # Forward pass
            outputs = model(frames.to(device), text.to(device), attention_mask.to(device), task)

            # Calculate the loss
            loss = criterion(outputs, label.to(device))
            text_loss.append(loss.item())

            preds = torch.argmax(outputs, dim=1)
            acc = np.mean(preds.detach().cpu().numpy() == label.detach().cpu().numpy())
            text_acc.append(acc)

        loss = sum(text_loss) / max(1,len(text_loss))
        acc = sum(text_acc) / max(1,len(text_acc))
        print(f'Epoch [{epoch+1}/{cfg.train.num_epochs}], Batch [{batch_idx}/{len(valid_contra_text_dataloader)}], Loss: {loss}, Acc: {acc}')
        wandb.log({'epoch': (epoch + 1) , 'batch': (batch_idx) / len(valid_contra_text_dataloader), 'valid_loss_text': loss, 'valid_acc_text': acc})

        for batch_idx, (clip1, clip2, text, attention_mask, label) in enumerate(valid_contra_video_dataloader):
            task = 'video_clf'
            clip1 = clip1[:,:,:,:2]
            clip2 = clip2[:,:,:,:2]
            b, s, k, d = clip1.shape
            if b == 1:
                continue
            frames = torch.cat([clip1.reshape(b, s, -1), clip2.reshape(b, s, -1)], dim=1)

            # Forward pass
            outputs = model(frames.to(device), text.to(device), attention_mask.to(device), task)

            # Calculate the loss
            loss = criterion(outputs, label.to(device))
            video_loss.append(loss.item())

            preds = torch.argmax(outputs, dim=1)
            acc = np.mean(preds.detach().cpu().numpy() == label.detach().cpu().numpy())
            video_acc.append(acc)

        loss = sum(video_loss) / max(1,len(video_loss))
        acc = sum(video_acc) / max(1,len(video_acc))
        print(f'Epoch [{epoch+1}/{cfg.train.num_epochs}], Batch [{batch_idx}/{len(valid_contra_video_dataloader)}], Loss: {loss}, Acc: {acc}')
        wandb.log({'epoch': (epoch + 1) , 'batch': (batch_idx) / len(valid_contra_video_dataloader), 'valid_loss_video': loss, 'valid_acc_video': acc})


        # Save the model checkpoint
        try:
            torch.save({'state_dict': model.state_dict(),
                        'hyperparameters': cfg,
                        }, f'{cfg.train.model_dir}/checkpoints/{cfg.data.eval_article}-model.pth')
        except:
            pass
        wandb.save(f'{cfg.train.model_dir}/checkpoints/{cfg.data.eval_article}-model.pth')

    mean_text_loss = sum(text_loss) / max(1, len(text_loss))
    mean_video_loss = sum(video_loss) / max(1, len(video_loss))
    print(f'Text Loss: {mean_text_loss}, Video Loss: {mean_video_loss}')

    detect_loss = []
    detect_acc = []
    align_loss = []
    align_acc = []

    # reset optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    model, optimizer = accelerator.prepare(model, optimizer)

    for epoch in range(cfg.finetune.num_epochs):
        detect_loss = []
        detect_acc = []
        align_loss = []
        align_acc = []
        model.train()

        for batch_idx, (frames, text, attention_mask, label, start_positions, end_positions) in enumerate(ft_text_dataloader):
            optimizer.zero_grad()

            task = 'text_span'
            frames = frames[:,:,:,:2]
            b, s, k, d = frames.shape
            if b == 1:
                continue
            frames = frames.reshape(b, s, -1)

            # Forward pass
            outputs = model(frames.to(device), text.to(device), attention_mask.to(device), task)#.permute(0, 2, 1)
            start_logits, end_logits = outputs.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)  # [batch_size, sequence_length]
            end_logits = end_logits.squeeze(-1)      # [batch_size, sequence_length]
            
            # Calculate the loss
            # print("Output and label shapes:", outputs.shape, label.shape)
            # token_loss = binary_criterion(outputs, label.to(device))
            # one_mask = label == 1
            # zero_mask = label == 0
            # one_masked_loss = token_loss * one_mask.to(device)
            # zero_masked_loss = token_loss * zero_mask.to(device)

            # loss = one_masked_loss.mean() + zero_masked_loss.mean()
            # print("Start and end positions:", start_positions, end_positions)
            start_loss = criterion(start_logits, start_positions.to(device))
            end_loss = criterion(end_logits, end_positions.to(device))
            loss = (start_loss + end_loss) / 2

            # print("losses:", start_loss, end_loss, loss)
            # print("preds:", torch.argmax(start_logits, dim=1), torch.argmax(end_logits, dim=1))
            # print("labels:", start_positions, end_positions)


            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            align_loss.append(loss.item())

            start_preds = torch.argmax(start_logits, dim=1)
            end_preds = torch.argmax(end_logits, dim=1)
            preds = torch.zeros_like(label)
            for i in range(len(start_preds)):
                preds[i, start_preds[i]:end_preds[i]+1] = 1
            acc = compute_iou(preds.detach().cpu().numpy(), label.detach().cpu().numpy())
            align_acc.append(acc)


        # Print the loss
        loss = sum(align_loss) / len(align_loss)
        # loss = 0
        acc = sum(align_acc) / len(align_acc)

        print(f'Epoch [{epoch+1}/{cfg.finetune.num_epochs}], Batch [{batch_idx}/{len(ft_text_dataloader)}], Loss: {loss}, Acc: {acc}')
        wandb.log({'epoch': (epoch + 1), 'batch': (batch_idx) / len(ft_text_dataloader), 'train_loss_align': loss, 'train_acc_align': acc})

        for batch_idx, (frames, text, attention_mask, label) in enumerate(ft_video_dataloader):
            optimizer.zero_grad()

            task = 'video_span'
            frames = frames[:,:,:,:2]
            b, s, k, d = frames.shape
            if b == 1:
                continue
            frames = frames.reshape(b, s, -1)

            # Forward pass
            outputs = model(frames.to(device), text.to(device), attention_mask.to(device), task).permute(0, 2, 1)

            # Calculate the loss
            token_loss = binary_criterion(outputs, label.to(device))
            one_mask = label == 1
            zero_mask = label == 0
            one_masked_loss = token_loss * one_mask.to(device)
            zero_masked_loss = token_loss * zero_mask.to(device)
            loss = one_masked_loss.mean() + zero_masked_loss.mean()

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            detect_loss.append(loss.item())

            preds = torch.argmax(outputs, dim=1)
            acc = compute_iou(preds.detach().cpu().numpy(), label.detach().cpu().numpy())
            detect_acc.append(acc)

        # Print the loss
        loss = sum(detect_loss) / len(detect_loss)
        # loss = 0
        acc = sum(detect_acc) / len(detect_acc)
        print(f'Epoch [{epoch+1}/{cfg.finetune.num_epochs}], Batch [{batch_idx}/{len(ft_video_dataloader)}], Loss: {loss}, Acc: {acc}')
        wandb.log({'epoch': (epoch + 1), 'batch': (batch_idx) / len(ft_video_dataloader), 'train_loss_detect': loss, 'train_acc_detect': acc})

        # Compute validation
        model.eval()
        detect_loss = []
        detect_acc = []
        detect_preds = []
        detect_labels = []
        align_loss = []
        align_acc = []
        align_preds = []
        align_labels = []

        for batch_idx, (frames, text, attention_mask, label, start_positions, end_positions) in enumerate(valid_ft_text_dataloader):
            optimizer.zero_grad()

            task = 'text_span'
            frames = frames[:,:,:,:2]
            b, s, k, d = frames.shape
            if b == 1:
                continue
            frames = frames.reshape(b, s, -1)

            # Forward pass
            outputs = model(frames.to(device), text.to(device), attention_mask.to(device), task)#.permute(0, 2, 1)
            start_logits, end_logits = outputs.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)  # [batch_size, sequence_length]
            end_logits = end_logits.squeeze(-1)      # [batch_size, sequence_length]
            
            # Calculate the loss
            # print("Output and label shapes:", outputs.shape, label.shape)
            # token_loss = binary_criterion(outputs, label.to(device))
            # one_mask = label == 1
            # zero_mask = label == 0
            # one_masked_loss = token_loss * one_mask.to(device)
            # zero_masked_loss = token_loss * zero_mask.to(device)

            # loss = one_masked_loss.mean() + zero_masked_loss.mean()

            start_loss = criterion(start_logits, start_positions.to(device))
            end_loss = criterion(end_logits, end_positions.to(device))
            loss = (start_loss + end_loss) / 2
            
            align_loss.append(loss.item())
            
            start_preds = torch.argmax(start_logits, dim=1)
            end_preds = torch.argmax(end_logits, dim=1)
            # print("Start and end preds:", start_preds, end_preds)

            preds = torch.zeros_like(label)
            for i in range(len(start_preds)):
                preds[i, start_preds[i]:end_preds[i]+1] = 1
            acc = compute_iou(preds.detach().cpu().numpy(), label.detach().cpu().numpy())
            align_acc.append(acc)

            for pred in preds:
                align_preds.append(list(map(lambda x: str(x.item()), pred)))
            for l in label:
                align_labels.append(list(map(lambda x: str(x.item()), l)))

        # Print the loss
        loss = sum(align_loss) / max(1, len(align_loss))
        # loss = 0
        acc = sum(align_acc) / max(1, len(align_acc))
        print(f'Epoch [{epoch+1}/{cfg.finetune.num_epochs}], Batch [{batch_idx}/{len(valid_ft_text_dataloader)}], Loss: {loss}, Acc: {acc}')
        wandb.log({'epoch': (epoch + 1), 'batch': (batch_idx) / len(valid_ft_text_dataloader), 'valid_loss_align': loss, 'valid_acc_align': acc})
        
        for batch_idx, (frames, text, attention_mask, label) in enumerate(valid_ft_video_dataloader):
            task = 'video_span'
            frames = frames[:,:,:,:2]
            b, s, k, d = frames.shape
            if b == 1:
                continue
            frames = frames.reshape(b, s, -1)

            # Forward pass
            outputs = model(frames.to(device), text.to(device), attention_mask.to(device), task).permute(0, 2, 1)

            # Calculate the loss
            token_loss = binary_criterion(outputs, label.to(device))
            one_mask = label == 1
            zero_mask = label == 0
            one_masked_loss = token_loss * one_mask.to(device)
            zero_masked_loss = token_loss * zero_mask.to(device)
            loss = one_masked_loss.mean() + zero_masked_loss.mean()
            
            detect_loss.append(loss.item())

            preds = torch.argmax(outputs, dim=1)
            acc = compute_iou(preds.detach().cpu().numpy(), label.detach().cpu().numpy())
            detect_acc.append(acc)

            for pred in preds:
                detect_preds.append(list(map(lambda x: str(x.item()), pred)))
            for l in label:
                detect_labels.append(list(map(lambda x: str(x.item()), l)))

        # Print the loss
        loss = sum(detect_loss) / len(detect_loss)
        # loss = 0
        acc = sum(detect_acc) / len(detect_acc)
        print(f'Epoch [{epoch+1}/{cfg.finetune.num_epochs}], Batch [{batch_idx}/{len(valid_ft_video_dataloader)}], Loss: {loss}, Acc: {acc}')
        wandb.log({'epoch': (epoch + 1), 'batch': (batch_idx) / len(valid_ft_video_dataloader), 'valid_loss_detect': loss, 'valid_acc_detect': acc})



        # Save the model checkpoint
        try:
            torch.save({'state_dict': model.state_dict(),
                        'hyperparameters': cfg,
                        }, f'{cfg.finetune.model_dir}/checkpoints/{cfg.data.eval_article}-model.pth')
        except:
            pass
        with open(f'{cfg.finetune.model_dir}/outputs/{cfg.data.eval_article}-detect_preds.txt', 'w') as f:
            for pred in detect_preds:
                f.write(','.join(pred) + '\n')
        with open(f'{cfg.finetune.model_dir}/outputs/{cfg.data.eval_article}-detect_labels.txt', 'w') as f:
            for label in detect_labels:
                f.write(','.join(label) + '\n')
        with open(f'{cfg.finetune.model_dir}/outputs/{cfg.data.eval_article}-align_preds.txt', 'w') as f:
            for pred in align_preds:
                f.write(','.join(pred) + '\n')
        with open(f'{cfg.finetune.model_dir}/outputs/{cfg.data.eval_article}-align_labels.txt', 'w') as f:
            for label in align_labels:
                f.write(','.join(label) + '\n')

        wandb.save(f'{cfg.finetune.model_dir}/checkpoints/{cfg.data.eval_article}-model.pth')
        wandb.save(f'{cfg.finetune.model_dir}/outputs/{cfg.data.eval_article}-detect_preds.txt')
        wandb.save(f'{cfg.finetune.model_dir}/outputs/{cfg.data.eval_article}-detect_labels.txt')
        wandb.save(f'{cfg.finetune.model_dir}/outputs/{cfg.data.eval_article}-align_preds.txt')
        wandb.save(f'{cfg.finetune.model_dir}/outputs/{cfg.data.eval_article}-align_labels.txt')

    # mean_detect_loss = sum(detect_loss) / len(detect_loss)
    # mean_align_loss = sum(align_loss) / len(align_loss)
    # print(f'Detect Loss: {mean_detect_loss}, Align Loss: {mean_align_loss}')
    wandb.finish()


    return 0

def train(cfg):
    if cfg.model.type == 'multitask':
        return multitask(cfg)
    elif cfg.model.type == 'singletask':
        return singletask(cfg)

if __name__ == '__main__':
    cfg = namedtuple('Config', ['seed', 'meta', 'data', 'model', 'train', 'wandb', 'finetune'])
    cfg.seed = 42
    cfg.meta = namedtuple('Meta', ['device'])
    cfg.meta.device = 'cuda'
    cfg.data = namedtuple('Data', ['datadir', 'label_file', 'clip_length', 'batch_size', 'ft_datadir', 'ft_label_file'])
    cfg.data.datadir = 'ASL_STEM_Wiki_data/videos/'
    cfg.data.label_file = 'ASL_STEM_Wiki_data/videos.csv'
    cfg.data.clip_length = 500 # 4 * clip_length is actual max video length
    cfg.data.batch_size = 4
    cfg.data.seq_length = 600
    cfg.data.ft_datadir = 'ASL_STEM_Wiki_data/videos/'
    cfg.data.ft_label_file = 'fs-annotations/train.csv'
    cfg.data.eval_article = 'Hal Anger'
    cfg.model = namedtuple('Model', ['type', 'num_keypoints', 'hidden_feature', 'p_dropout', 'num_stages'])
    cfg.model.type = 'multitask'
    cfg.model.num_keypoints = 75
    cfg.model.hidden_feature = 768 # try smaller hidden size
    cfg.model.p_dropout = 0.3
    cfg.model.num_stages = 6
    cfg.train = namedtuple('Train', ['learning_rate', 'num_epochs', 'model_dir'])
    cfg.train.learning_rate = 1e-3
    cfg.train.num_epochs = 0
    cfg.train.model_dir = 'test'
    cfg.wandb = namedtuple('Wandb', ['name'])
    cfg.wandb.name = 'st-Hal_Anger'
    cfg.finetune = namedtuple('Finetune', ['num_epochs', 'model_dir'])
    cfg.finetune.num_epochs = 40
    cfg.finetune.model_dir = 'test'

    loss = train(cfg)
    print(loss)
    



