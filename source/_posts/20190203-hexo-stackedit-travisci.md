---
title:  StackEdit 在线写作，Travis-CI 自动部署实现
---

> 之前用Hexo + GitHub Pages 搭建过博客，写了几篇就弃坑了。


大概流程：
 - 将配置好的Hexo 博客推送到自己Github repo内；
 - Travis-CI 与Github账号关联，激活博客repo；
 - StackEdit 写博客，push 到 博客repo；
 - Travis-CI 检测到博客 repo有更新，拉取下来，安装编译，执行`hexo d -g` 部署更新到Github Pages。

我自己的情况是文章推送到repo 的 master 分支，部署时推送到publish分支以供Github Pages展示。

## 零、准备工作
 
准备工作完成之后，得到：

 - 在github为博客创建的repo
 - 配置repo的deploy key
 - hexo 安装配置完成（初次使用记得添加CNAME文件到source目录）
 - 安装ruby，最好是最新的稳定版

 
## 一、在线写作

> 解决方案：StackEdit

不想注册什么在线写作平台，本来想自己写一个简单页面来写markdown的，碰巧在github上搜 `markdown editor`，第一个结果就是 [stackedit](https://stackedit.io/)，点进去一看，支持同步到github，正好符合我的需求。

## 二、自动部署

> 解决方案： Travis-CI

[Travis-CI](https://www.travis-ci.org/) 是一个持续集成测试的工具，Github上public repo使用是免费的，这里不做介绍了。

Travis-CI 的配置：

略，参考网上相关教程（如 [用Travis CI 自动部署 Hexo](https://segmentfault.com/a/1190000004667156)）。

## 三、遇到的坑

加密 deploy key 私钥时，需要用到gem 安装 travis-ci cli工具：

 - `$ gem install travis`

因为我本地电脑不方便，直接上服务器整了，服务器的系统是centos，用官方提供的2.0.0的ruby，travis 直接装不上，重新装了个2.5.1，这才装好。
附[centos安装ruby教程](https://linuxize.com/post/how-to-install-ruby-on-centos-7/)。

