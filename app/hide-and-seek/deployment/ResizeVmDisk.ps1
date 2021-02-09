# Run this from your local machine via PowerShell.
# Azure PowerShell is required: https://docs.microsoft.com/en-us/powershell/azure/install-az-ps?view=azps-5.0.0
param ($paramVmName, [boolean] $login=$false)
if ($login -eq $true) 
{
    Connect-AzAccount
}
Select-AzSubscription -SubscriptionName 'Microsoft Azure Sponsorship 2'
$rgName = 'hide-and-seek'
$vmName = $paramVmName
$vm = Get-AzVM -ResourceGroupName $rgName -Name $vmName
Stop-AzVM -Force -ResourceGroupName $rgName -Name $vmName
$disk= Get-AzDisk -ResourceGroupName $rgName -DiskName $vm.StorageProfile.OsDisk.Name
$disk.DiskSizeGB = 1023
Update-AzDisk -ResourceGroupName $rgName -Disk $disk -DiskName $disk.Name
Start-AzVM -ResourceGroupName $rgName -Name $vmName
write-host "Done"