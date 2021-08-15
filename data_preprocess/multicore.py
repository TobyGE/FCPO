from read_write import check_folder_exist, write_lines
import numpy as np

def run_multicore(data, save_path, u_core = 10, i_core = 10):
    '''
    Encode and save multi-core data
    
    @input:
    - data: [(uid, iid, ...)]
    - save_path: target file directory to save filtered multicore data
    - n_core: core size

    @save files:
    - save_path/multicore_[n_core].csv(the encoded file): [(encoded uid, encoded iid, rating)]

    '''
    users = []
    uMap = dict()
    uCounts = {}
    items = []
    iMap = dict()
    iCounts = {}
    for i in range(len(data)):
        u = data[i][0]
        v = data[i][1]
        # For each unique user, find its encoded id and count
        if u not in uMap:
            uMap[u] = len(users)
            users.append(u)
            uCounts[u] = 1
        else:
            uCounts[u] += 1
        # For each unique item, find its encoded id and count
        if v not in iMap:
            iMap[v] = len(items)
            items.append(v)
            iCounts[v] = 1
        else:
            iCounts[v] += 1

    # filter multi core
    print("Filtering " + str(u_core) + "_" + str(i_core) + "-core data")
    iteration = 0
    lastChange = np.float("inf")  # the number of removed record
    proposedData = data
#         print("Original number of records: " + str(len(domainData)))
    while lastChange != 0:
        iteration += 1
        print("Iteration " + str(iteration))
        changeNum = 0
        newData = []
        # each iteration, count number of records that need to delete
        for record in proposedData:
            user = record[0]
            item = record[1]
            if uCounts[user] < u_core or iCounts[item] < i_core:
                uCounts[user] -= 1
                iCounts[item] -= 1
                changeNum += 1
            else:
                newData.append(record)
        proposedData = newData
        print("Number of removed record: " + str(changeNum))
        if changeNum > lastChange:
            print("Not converging")
            proposedData = domainData
            break
        lastChange = changeNum
    
    proposedData = np.stack(proposedData)
    user = np.unique(proposedData[:,0])
    item = np.unique(proposedData[:,1])
    userMap = {user[i]: i for i in range(len(user))}
    itemMap = {item[i]: i for i in range(len(item))}
    # encode user id and item id
    encodedData = []
    for record in proposedData:
        encodedData.append([int(userMap[record[0]]), int(itemMap[record[1]]), record[2], record[3]])
#     print("Number of records after filtering: " + str(len(encodedData)) + "/" + str(len(domainData)))
    write_lines(save_path + "/multicore_" + str(u_core) + "_" + str(i_core) + ".csv", encodedData)
    return encodedData


def run_domain_multicore(data, save_path, n_core = 10, auto_core = False, filter_rate = 0.2):
    '''
    Encode and save multi core data
    @save files:
    - save_path/multicore_***/domain.csv(the encoded file): [(encoded uid, encoded iid, rating)]
    '''
    if auto_core:
        outPath = save_path + "multicore_auto/"
    else:
        outPath = save_path + "multicore" + str(n_core) + "/"
    check_folder_exist(outPath)
    
    # user ids shared across
    users = []
    uMap = dict()
    uCounts = {}
    for domain, domainData in data.items():
        print("Filtering domain: " + domain)
        items = []
        iMap = dict()
        iCounts = {}
        for i in range(len(domainData)):
            u = domainData[i][0]
            v = domainData[i][1]
            # For each unique user, find its encoded id and count
            if u not in uMap:
                uMap[u] = len(users)
                users.append(u)
                uCounts[u] = 1
            else:
                uCounts[u] += 1
            # For each unique item, find its encoded id and count
            if v not in iMap:
                iMap[v] = len(items)
                items.append(v)
                iCounts[v] = 1
            else:
                iCounts[v] += 1
        if auto_core:
            print("Automatically find n_core that filter " + str(100*filter_rate) + "% of item")
            nCoreCounts = dict()
            for v,c in iCounts.items():
                if c not in nCoreCounts:
                    nCoreCounts[c] = [0,1]
                else:
                    nCoreCounts[c][1] += 1
            for u,c in uCounts.items():
                if c not in nCoreCounts:
                    nCoreCounts[c] = [1,0]
                else:
                    nCoreCounts[c][0] += 1
                
            userToRemove = 0
            itemToRemove = 0
            for c,counts in sorted(nCoreCounts.items()):
                userToRemove += counts[0] * c
                itemToRemove += counts[1] * c
#                 print("Remove (" + str(userToRemove) + "," + str(itemToRemove) + ") when n_core = " + str(c))
                if userToRemove > filter_rate * len(domainData) or itemToRemove > filter_rate * len(domainData):
                    n_core = c + 1
                    break
        # filter multi core
        print("Filtering " + str(n_core-1) + "-core data")
        iteration = 0
        lastChange = np.float("inf")  # the number of removed record
        proposedData = domainData
#         print("Original number of records: " + str(len(domainData)))
        while lastChange != 0:
            iteration += 1
            print("Iteration " + str(iteration))
            changeNum = 0
            newData = []
            # each iteration, count number of records that need to delete
            for record in proposedData:
                user = record[0]
                item = record[1]
                if uCounts[user] < n_core or iCounts[item] < n_core:
                    uCounts[user] -= 1
                    iCounts[item] -= 1
                    changeNum += 1
                else:
                    newData.append(record)
            proposedData = newData
            print("Number of removed record: " + str(changeNum))
            if changeNum > lastChange:
                print("Not converging")
                proposedData = domainData
                break
            lastChange = changeNum

        # encode user id and item id
        encodedData = []
        for record in proposedData:
            encodedData.append([uMap[record[0]], iMap[record[1]], record[2]])
        print("Number of records after filtering: " + str(len(encodedData)) + "/" + str(len(domainData)))
        
        # save files
        write_lines(outPath + domain + ".csv", encodedData)
        return encodedData