### 🎓 Masters thesis 🎓

#### Michal Glos (xglosm01)
#### FIT - VUTBR

#### Playing football in the Google football environment [gfootball](https://gitlab.com/michal.glos99/dip/-/tree/main)

#### Source code:
##### Setup:
 - [Install docker](https://docs.docker.com/engine/install/)
 - [Install nvidia-containers](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
 - In the root of the project, you coud leverage Makefile and it's command `make`:
	 - `make build` build *gfootball* container with GPU support
	 - `make start` start *gfootball* container with GPU support and connected DISPLAY
	 - `make play` run Google football in *gfootball*  container 

##### Relevant sources:
 - https://arxiv.org/abs/2101.10382 (Curriculum learning survey 2021 (april 2022 published))
 - https://arxiv.org/pdf/2210.17368.pdf (Master thesis (2022) - tranfer learning) - good source for general information about RL (til curriculum learning), have to cite the original source tho
 - https://theses.hal.science/tel-03633787/document Disertation thesis - quite detailed description of the same topics as masters thesis above, also contains Automated curriculum learning
 - https://arxiv.org/pdf/2003.04960.pdf curriculum learning applied to reinforced learning ... that's ... that's .. it?
 - https://openreview.net/pdf?id=_cFdPHRLuJ Curriculum learning gradual domain adaptation
 - https://arxiv.org/pdf/2206.07505.pdf Agent cooperation benchmark
 - https://openreview.net/pdf?id=BkggGREKvS CoachReg complete paper

##### Errors encountered:
Error 1:
 - 	```
 	docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
	ERRO[0000] error waiting for container: context canceled
	```
- Solution 1:
 -  ```
	distribution=$(. /etc/os-release;echo $ID$VERSION_ID)       && curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -       && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
	sudo apt-get update
	sudo apt-get install -y nvidia-docker2
	sudo systemctl restart docker
	```
