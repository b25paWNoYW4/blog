---
title: Laradock环境搭建
categories: Docker
date: 2017-12-17 22:40:03
tags:
---

## 准备工作

 1.   系统：Centos7.3 64位

        df -h //查看系统磁盘空间情况  
        dh -sh //查看文件夹空间占用情况  
        scp -r d://.../Laravel root@111.22.333.44:/home  //上传文件到服务器中  

 2.   下载laradock到项目目录内。

        $ git clone https://github.com/Laradock/laradock.git
 
## 安装   
 照着官方文档，进入laradock目录后：

        $ cp env-example .env

        $ docker-compose up -d nginx mysql
 
 <!-- more -->

 安装Node和npm（或者Node+yarn）：
 
 打开docker-compose.yml ,搜索 `INSTALL_NODE`，将其设置为 `true`。

        workspace:
                build:docker-compose down
                    context: ./workspace
                    args:
                        + INSTALL_NODE=true
                    // 安装yarn，还要设置INSTALL_YARN=true
        ... 

 然后重新编译容器：

        $ docker-compose build workspace

 最好再重启一下：

        $ docker-compose down
        $ docker-compose up -d nginx mysql  

 

## 配置安装

 6.   进入到项目目录中，修改 `.env`文件：
        
        DB_HOST=mysql //其它数据库的用户名密码记得设置对

    [创建数据库用户](#create)

 7.   修改cache文件夹权限：
        
        $ sudo chmod -R 777 storage bootstrap/cache

 8.   进入workspace：
               
        $ docker-compose exec workspace bash //默认以root用户进入，不推荐
        $ docker-compose exec --user=laradock workspace bash //推荐

 4.   执行：

        $ npm install && composer install

## 结束

 现在访问 localhost ，应该能看到些东西了。

<h2 id="create">数据库</h2>

 进入数据库容器：

        $ docker-compose exec mysql bash
        $ mysql -u root -p
        $ enter passwd:             //默认是root
创建用户：

        $ CREATE USER 'username'@'localhost' IDENTIFIED BY 'password';
赋予权限：
        
        $ GRANT ALL ON *.* TO 'username'@'localhost';

## 错误

 - Undefined property: stdClass::$column_name when generating a model
 
 解决办法：

        $ sudo vi /vendor/laravel/framework/src/Illuminate/Database/Schema/Grammars/MySqlGrammar.php

 找到 `select column_name from information_schema.columns`

 修改为：

        select column_name as `column_name` from information_schema.columns

 参考链接：  

 [1.https://github.com/laravel/framework/issues/20190](https://github.com/laravel/framework/issues/20190)  

 [2.http://laradock.io/introduction/](http://laradock.io/introduction/)
