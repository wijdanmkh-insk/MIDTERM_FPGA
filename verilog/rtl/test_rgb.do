transcript on

if {[file exists work]} {
    vdel -lib work -all
}

vlib work

vlog -sv cnn_top_rgb.v
vlog -sv tb_cnn_top_rgb.v

vsim -voptargs=+acc work.tb_cnn_top

run -all


quit -f
