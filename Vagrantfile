# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "precise64"

  config.vm.box_url = "http://files.vagrantup.com/precise64.box"

  config.vm.synced_folder "/Volumes/Imagens/Datasets/", "/home/guilherme/images"
  config.vm.synced_folder "/Users/guilherme/code", "/home/guilherme/code"

  config.vm.provider :virtualbox do |vb|
    vb.customize ['modifyvm', :id, '--usb', 'on']
    vb.customize ["modifyvm", :id, "--usbehci", "on"]
    vb.customize ["modifyvm", :id, "--memory", "2048"]
    vb.customize ["modifyvm", :id, "--cpus", "2"]
    vb.customize ["modifyvm", :id, "--ioapic", "on"]
    #vb.gui = true
  end

end
