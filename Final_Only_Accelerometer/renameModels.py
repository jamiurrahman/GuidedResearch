'''
import os


for count, filename in enumerate(os.listdir("./SavedModel/20200408-231907")):
    # dst = "Hostel" + str(count) + ".jpg"
    # src = 'xyz' + filename
    # dst = 'xyz' + dst

    # print(f"{count} : {filename}")
    if filename.__contains__(".h5"):
        counter = int(filename.split("_")[0])
        lastNumber = filename.split("_")[-1]

        filename = "./SavedModel/20200408-231907/" + filename

        for i in range(1,50):
            if ((counter == 26) or (counter == (26 + (40 * i)))):
                # print(f"{counter} : {lastNumber}")
                print("Before : ", filename)
                dst_filename = filename.replace(lastNumber, "32.h5")
                print("After : ", dst_filename)
                os.rename(filename, dst_filename)
                break

            elif ((counter == 27) or (counter == (27 + (40 * i)))):
                # print(f"{counter} : {lastNumber}")
                print("Before : ", filename)
                dst_filename = filename.replace(lastNumber, "64.h5")
                print("After : ", dst_filename)
                os.rename(filename, dst_filename)
                break
            elif ((counter == 28) or (counter == (28 + (40 * i)))):
                # print(f"{counter} : {lastNumber}")
                print("Before : ", filename)
                dst_filename = filename.replace(lastNumber, "128.h5")
                print("After : ", dst_filename)
                os.rename(filename, dst_filename)
                break

            elif ((counter == 29) or (counter == (29 + (40 * i)))):
                # print(f"{counter} : {lastNumber}")
                print("Before : ", filename)
                dst_filename = filename.replace(lastNumber, "256.h5")
                print("After : ", dst_filename)
                os.rename(filename, dst_filename)
                break

            elif ((counter == 30) or (counter == (30 + (40 * i)))):
                # print(f"{counter} : {lastNumber}")
                print("Before : ", filename)
                dst_filename = filename.replace(lastNumber, "512.h5")
                print("After : ", dst_filename)
                os.rename(filename, dst_filename)
                break

            elif ((counter == 31) or (counter == (31 + (40 * i)))):
                # print(f"{counter} : {lastNumber}")
                print("Before : ", filename)
                dst_filename = filename.replace(lastNumber, "32.h5")
                print("After : ", dst_filename)
                os.rename(filename, dst_filename)
                break

            elif ((counter == 32) or (counter == (32 + (40 * i)))):
                # print(f"{counter} : {lastNumber}")
                print("Before : ", filename)
                dst_filename = filename.replace(lastNumber, "64.h5")
                print("After : ", dst_filename)
                os.rename(filename, dst_filename)
                break
            elif ((counter == 33) or (counter == (33 + (40 * i)))):
                # print(f"{counter} : {lastNumber}")
                print("Before : ", filename)
                dst_filename = filename.replace(lastNumber, "128.h5")
                print("After : ", dst_filename)
                os.rename(filename, dst_filename)
                break

            elif ((counter == 34) or (counter == (34 + (40 * i)))):
                # print(f"{counter} : {lastNumber}")
                print("Before : ", filename)
                dst_filename = filename.replace(lastNumber, "256.h5")
                print("After : ", dst_filename)
                os.rename(filename, dst_filename)
                break

            elif ((counter == 35) or (counter == (35 + (40 * i)))):
                # print(f"{counter} : {lastNumber}")
                print("Before : ", filename)
                dst_filename = filename.replace(lastNumber, "512.h5")
                print("After : ", dst_filename)
                os.rename(filename, dst_filename)
                break

            elif ((counter == 36) or (counter == (36 + (40 * i)))):
                # print(f"{counter} : {lastNumber}")
                print("Before : ", filename)
                dst_filename = filename.replace(lastNumber, "32.h5")
                print("After : ", dst_filename)
                os.rename(filename, dst_filename)
                break

            elif ((counter == 37) or (counter == (37 + (40 * i)))):
                # print(f"{counter} : {lastNumber}")
                print("Before : ", filename)
                dst_filename = filename.replace(lastNumber, "64.h5")
                print("After : ", dst_filename)
                os.rename(filename, dst_filename)
                break
            elif ((counter == 38) or (counter == (38 + (40 * i)))):
                # print(f"{counter} : {lastNumber}")
                print("Before : ", filename)
                dst_filename = filename.replace(lastNumber, "128.h5")
                print("After : ", dst_filename)
                os.rename(filename, dst_filename)
                break

            elif ((counter == 39) or (counter == (39 + (40 * i)))):
                # print(f"{counter} : {lastNumber}")
                print("Before : ", filename)
                dst_filename = filename.replace(lastNumber, "256.h5")
                print("After : ", dst_filename)
                os.rename(filename, dst_filename)
                break

            elif ((counter == 40) or (counter == (40 + (40 * i)))):
                # print(f"{counter} : {lastNumber}")
                print("Before : ", filename)
                dst_filename = filename.replace(lastNumber, "512.h5")
                print("After : ", dst_filename)
                os.rename(filename, dst_filename)
                break
'''