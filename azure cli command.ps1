az -h
az provider register -n Microsoft.Batch
az provider register -n Microsoft.BatchAI
az role assignment create --scope /subscriptions/1aa15964-43e9-4fab-9be5-81abdcb9c8d1 --role "Network Contributor" --assignee 9fcb3732-5f52-4135-8c08-9d4bbaf203ea
az batchai cluster create -h
az group create -l eastus -n holkrazure
### az batchai cluster create -l eastus -g holkrazure -n demoCluster -s Standard_NC6 -i UbuntuLTS --min 1 --max 1 -u demoUser -p demoPassword


### Storage 생성
az storage account create -l eastus -g holkrazure -n aitrainigstorageaz storage share create -n aiafs --account-name aitrainigstorageaz storage container create -n aicontainer --account-name aitrainigstorage