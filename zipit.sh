dir="log_espo_cd"
rm -rf $dir && cp -r log/ $dir && zip -rq $dir.zip $dir
sz $dir.zip
