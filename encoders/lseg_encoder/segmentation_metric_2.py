import torch
from argparse import ArgumentParser
import sys
import os
import tqdm

def calculate_iou(pred, target, num_classes):
    device = pred.device
    target = target.to(device)
    
    ious = []
    for c in range(num_classes):
        TP = torch.sum((pred == c) & (target == c)).item()
        FP = torch.sum((pred == c) & (target != c)).item()
        FN = torch.sum((pred != c) & (target == c)).item()
        
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else float('nan')
        ious.append(iou)
    
    mIoU = torch.nanmean(torch.tensor(ious)).item()
    return mIoU

def calculate_accuracy(pred, target):
    device = pred.device
    target = target.to(device)
    
    correct = torch.sum(pred == target).item()
    total = target.numel()
    accuracy = correct / total
    return accuracy

def get_label_dataset(label_dir):
    label_files = []
    for file_name in os.listdir(label_dir):
        if file_name.endswith("_label_CxHxW.pt"):
            label_files.append(os.path.join(label_dir, file_name))
    label_files.sort()
    return label_files


def test(args, llffhold=8):
    testset_student = get_label_dataset(args.student_label_dir)
    testset_teacher = get_label_dataset(args.teacher_label_dir)
    
    if args.eval_mode == 'test':
        testset_teacher = [c for idx, c in enumerate(testset_teacher) if idx % llffhold == 2]

    tbar = tqdm.tqdm(zip(testset_student, testset_teacher), total=len(testset_student))

    iou_accum = 0
    accuracy_accum = 0
    count = 0

    num_classes = 66

    for i, (student_file, teacher_file) in enumerate(tbar):
        student_label = torch.load(student_file, weights_only=False)
        teacher_label = torch.load(teacher_file, weights_only=False)

        device = student_label.device
        teacher_label = teacher_label.to(device)

        # mIoUの計算
        iou = calculate_iou(student_label, teacher_label, num_classes)

        # Accuracyの計算
        accuracy = calculate_accuracy(student_label, teacher_label)

        iou_accum += iou
        accuracy_accum += accuracy
        count += 1

        # result
        tbar.write(f"for the {i}th image, the accuracy: {accuracy:.4f}, iou: {iou:.4f}")

    # final result
    print(f"Average Accuracy: {accuracy_accum/count:.4f}, Average IoU: {iou_accum/count:.4f}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--teacher-label-dir', required=True, help="教師モデルのラベルディレクトリ")
    parser.add_argument('--student-label-dir', required=True, help="学生モデルのラベルディレクトリ")
    parser.add_argument('--eval-mode', required=True, help="評価モード")

    args = parser.parse_args(sys.argv[1:])
    
    test(args)
