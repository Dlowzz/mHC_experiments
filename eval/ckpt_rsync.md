rsync -avP root@10.224.62.2:/home/work/data/guotianzizhe02/data/ckpt/ /home/work/data/guotianzizhe/data/test/
rsync -avP root@10.224.62.5:/home/work/data/test/testckpt/ /home/work/data/guotianzizhe/data/test/
rsync -avP -e "ssh -p 35242"
  root@ssh-cn-huabei1.ebcloud.com:/home/work/data/test/testckpt/
  /home/work/data/guotianzizhe/data/test/
