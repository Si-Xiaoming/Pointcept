构建docker
```shell
docker build -t image_name -f Dockerfile .
```
docker build -t ptcept:01 -f Dockerfile -build_arg HTTPS_PROXY=127.0.0.1:10809 HTTP_PROXY=127.0.0.1:10809 

docker 创建容器
```shell
docker run -it --gpus all -m 30g --cpus 8 ^
    -v %CD%:/workspace ^
    -p 8000:22 ^
    cept:02
```
>交互式终端模式（
-i
保持输入，
-t
分配伪终端）


将本地的docker容器推送到docker hub
```
docker push 
```
todo install ssh
```
apt-get install -y openssh-server
service ssh start

```
RUN apt-get update && apt-get install -y openssh-server
CMD ["/usr/sbin/sshd", "-D"]

docker exec -it /bin/bash ID