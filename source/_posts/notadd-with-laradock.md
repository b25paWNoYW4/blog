---
title: unrecognized import path "golang.org/x/sys/unix"
categories: Docker
date: 2017-10-25 14:37:02
---

 服务器环境：  
 **Centos7 内核：4.13.4-1.el7.elrepo.x86_64**  **Docker 17.06.2-ce, build cec0b72**

 前面照着Notadd的[教程](https://docs.notadd.com/laradock/)来  
 到 `docker-compose up caddy postgres pgadmin` 这一步时，会遇到：

        package golang.org/x/sys/unix: unrecognized import path "golang.org/x/sys/unix" (https fetch: Get https://golang.org/x/sys/unix?go-get=1: dial tcp 216.239.37.1:443: i/o timeout)

        ERROR: Service 'caddy' failed to build: The command '/bin/sh -c go get -v github.com/abiosoft/caddyplug/caddyplug     && caddyplug install-caddy     && caddyplug install git' returned a non-zero code: 1

<!-- more -->

 这是因为golang的服务器不在国内。  

 解决办法是：

        $ sudo vi caddy/Dockerfile 

 在：

        RUN go get -v github.com/abiosoft/caddyplug/caddyplug \

 前加上：

        RUN mkdir -p $GOPATH/src/golang.org/x/
        RUN cd $GOPATH/src/golang.org/x/ \
            && git clone https://github.com/golang/sys.git sys

 然后重新执行 `docker-compose up caddy postgres pgadmin`，完成安装。

 参考链接：  
 [https://my.oschina.net/xxbAndy/blog/846722](https://my.oschina.net/xxbAndy/blog/846722)


