Code borrows from DeblurGANv2, EDSR, and SRCNN.

```bash

python train.py --train-file /media/qian/7f6908d4-b97f-4a1e-ba90-d502c5308801/DIV2K_train_HR --eval-file /media/qian/7f6908d4-b97f-4a1e-ba90-d502c5308801/DIV2K_valid_HR --outputs-dir outputs/edsr_test --logs_dir runs/edsr_test --model EDSR --criterion l1+perceptual --lr 2e-5 --batch-size 2 --num-epochs 3 --num-workers 4 --seed 123 --n_photon 1000 --f_num 16,20 --kernel jinc --num-channels 2
python eval.py --eval-file /media/qian/7f6908d4-b97f-4a1e-ba90-d502c5308801/DIV2K_valid_HR --model EDSR --model_path outputs/edsr_test/1000x16,20xjinc --seed 123 --n_photon 1000 --f_num 16,20 --kernel jinc --num-channels 2 --is_pred
python test.py --weights-file outputs/edsr_test/1000x16,20xjinc --image-file /media/qian/7f6908d4-b97f-4a1e-ba90-d502c5308801/DIV2K_valid_HR --output-path test/edsr_test --model EDSR --f_num 16,20 --n_photon 1000 --kernel jinc --num-channels 2

```
-----------------
#### DeblurGANv2

#### EDSR

#### SRCNN
