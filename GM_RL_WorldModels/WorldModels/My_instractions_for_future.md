
bash 01_generate_data.sh car_racing 4 250 300 0 5

# пришлось добавить несколько костылей

python3 02_train_vae.py --new_model 

# при запуске на обучение свежей модели, а при дообучениии без "--new_model".
# сделал так чтобы можно было дообучать пачками по 5, а то не хватало памяти
# модель не дообучалась (ValueError: Unable to load weights saved in HDF5 format into a subclassed Model which has not created its variables yet. Call the Model first, then load the weights.) нужно изменить формат файла на TF (было написано в интернете), но я просто прописал model.bult = True и заработало

python3 02_train_vae.py --k K = {0, 1, 2, 3, . . . 1000/5}


0) при начале работы

conda activate worldmodels

1) собрать данные просимулировав много раз

bash 01_generate_data.sh car_racing 4 250 300 0 5

2) обучить на данных вариационный энкодер на Й-той пачке данных (пачки размера 5, чтобы ноут тянул, но можно увеличить)

python3 02_train_vae.py --k Й

3) собрать данные для RNN

python3 03_generate_rnn_data.py

4) обучить RNN

python3 04_train_rnn.py --N 1000 --new_model --steps 100 --batch_size 32

python3 04_train_rnn.py --N 1000 --steps 32 --batch_size 100

python3 04_train_rnn.py --N 1000 --steps 100 --batch_size 100

5) обучить контроллер *****************

были проблемы с conda install -c conda-forge mpi4py
получилось после того как сначала установил sudo apt-get install openmpi-bin openmpi-common openssh-client openssh-server libopenmpi-dev , а потом установил conda install gxx_linux-64 потому что mpicc не находил компилятор

теперь есть mpi4py-3.0.3

и еще кое какие модули пришлось доустонавливать

xvfb-run -- python3 05_train_controller.py car_racing -n 2 -t 2 -e 4 --max_length 100

[ xvfb-run -- python3 05_train_controller.py car_racing -n 3 -t 3 -e 4 --max_length 500 ]

6) xvfb-run python 05_train_controller.py car_racing -n 4 -t 2 -e 3 --max_length 1000 --dream_mode 1


-[установка анаконды](https://phoenixnap.com/kb/how-to-install-anaconda-ubuntu-18-04-or-20-04)

-[виртуальное окружение](https://phoenixnap.com/kb/how-to-install-anaconda-ubuntu-18-04-or-20-04) не донастроил

-[виртуальное окружение](https://medium.com/@aaditya.chhabra/virtualenv-with-virtualenvwrapper-on-ubuntu-34850ab9e765)

-[open mpi](https://fnjn.io/2019/03/04/Install-OpenMPI-on-Ubuntu/)

-[mpi4py](https://pypi.org/project/mpi4py/)

-[Xvfb](https://linuxhint.com/install-xvfb-ubuntu/)












