Spinning Up:
1. Create conda environment:

2. Install pytorch in conda environment:
conda install pytorch torchvision torchaudio pytorch-cuda=11.1 -c pytorch -c nvidia

3. Install everything in the requirements using pip

4. Download https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth to ./

5. Create a folder called data_root.
Download and extract https://datasets.simula.no/downloads/kvasir-seg.zip to data_root
Download and extract https://www.dropbox.com/s/p5qe9eotetjnbmq/CVC-ClinicDB.rar?dl=1 to data_root

Trainings:
There are three jupyter files: teacher_train, student_train_alone, student_train_with_teacher
teacher_train -- used to train the teacher
student_train_alone -- used to train the student alone
student_train_with_teacher -- used to train the student with a teacher

Acknowledgements:
Our work forks the basic training loops used here:
https://github.com/ESandML/FCBFormer