set bench aes_cipher_top
open_lib ../Icc2Ndm/${bench}_nlib
copy_block -from_block ${bench} -to_block ${bench}_eco
open_block ${bench}_eco
link_block

save_block
exit