# 构建docker

一般使用Dockerfile构建docker的image，终端切到Dockerfile的路径

```shell
docker build -t image_name -f Dockerfile .
```
> docker build -t ptcept:01 -f Dockerfile --build-arg HTTPS_PROXY="http://host.docker.internal:10809" --build-arg HTTP_PROXY="http://host.docker.internal:10809" .
>
> docker build -t ptcept:01 -f Dockerfile .

创建好image后，需要创建对应的运行容器
```shell
docker run -it --gpus all -m 30g --cpus 8 ^
    -v %CD%:/workspace ^
    -p 8000:22 ^
    cept:02
```

>交互式终端模式（-i  保持输入，-t分配伪终端）
>
>-v 将该终端的路径挂载到docker容器中/workspace的路径下
>
>-p 将docker容器的22端口映射到宿主机的8000端口
>
>该容器是依据cept:02这个image创建




将本地的docker容器推送到docker hub
```
docker push 
```




# 释放docker_data空间

Docker Desktop的虚拟磁盘（如ext4.vhdx、docker_data.vhdx）在Windows系统中不会自动缩减空间占用。即使实际使用空间远低于最大值，系统仍会保留虚拟磁盘曾经达到的最大容量。

Docker Desktop基于WSL 2运行，而WSL 2使用vhdx格式的虚拟磁盘文件。这种文件支持自动扩容，但不会自动缩容。即使删除容器或镜像，虚拟磁盘文件仍会保留其最大占用空间，导致磁盘空间未被释放‌。

因此在删除需要自己释放空间：

```powershell
wsl --shutdown	#关闭WSL
diskpart		#打开diskpart模式
select vdisk file="<you_path>\docker\disk\docker_data.vhdx"
compact vdisk	#执行压缩
```



# Appendix

todo install ssh

```
apt-get install -y openssh-server
service ssh start

```

RUN apt-get update && apt-get install -y openssh-server
CMD ["/usr/sbin/sshd", "-D"]

docker exec -it /bin/bash ID