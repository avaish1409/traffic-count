import numpy as np
import cv2
import pandas as pd

from google.colab import drive
drive.mount('/content/gdrive')

import skimage.measure as measure

def check(x, y, eq):
  res = eq[0]*x + eq[1]*y + eq[2]
  org = eq[2]
  if (org>0) ^ (res>0):
    return False
  return True

def getEq(p1, p2):
  m = (p1[1]-p2[1])/(p1[0]-p2[0])
  c = (m*p2[0])-p1[1]
  return [-1*m, 1, c]

def fun(path, linexpos = 0, lineypos = 225, linexpos2 = 900, lineypos2 = 225):

  cap = cv2.VideoCapture(path)
  frames_count, fps, w, h = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
      cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  
  height = int(h)
  width = int(w)
  print('Input Video summary: ', frames_count, fps, width, height)

  df = pd.DataFrame(index=range(int(frames_count))) # dataframe of input video frames
  df.index.name = "Frames"

  # display variables
  framenumber = 0  
  carscrossedup = 0  
  carscrosseddown = 0  
  carids = []  
  caridscrossed = []  
  totalcars = 0  

  # processing - create background subtractor
  fgbg = cv2.createBackgroundSubtractorMOG2()


  # output video
  ret, frame = cap.read()
  ratio = .5  # resize ratio
  image = cv2.resize(frame, (0, 0), None, ratio, ratio) 
  width2, height2, channels = image.shape
  video = cv2.VideoWriter('tcount2-inpaint-mask-gray-inv.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (height2, width2), 1)
  
  # check night or day
  is_light = np.mean(image) > 117

  # equation of line
  eq = getEq((linexpos, lineypos), (linexpos2, lineypos2))

  print('Main task initiated')

  while True and framenumber < 100:

      ret, frame = cap.read()

      # check if input
      if ret:

          image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image

          gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray
          if not is_light:
            # gray[gray>200] = 100
            # gray = cv2.bitwise_not(gray)
            # inpaintMask = create_mask(image)
            (thresh, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            gray = cv2.inpaint( gray, im_bw, 10, cv2.INPAINT_TELEA)
            gray = cv2.bitwise_not(gray)

          fgmask = fgbg.apply(gray)  # uses the background subtraction
          # applies different thresholds to fgmask to try and isolate cars
          kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
          closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
          opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
          dilation = cv2.dilate(opening, kernel)
          

          retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)  # removes the shadows

          # if is_light:
          #   retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)  # removes the shadows
          # else:
          #   retvalbin, bins = cv2.threshold(dilation, 50, 255, cv2.THRESH_BINARY_INV)  # removes the car lights
            # bins = cv2.adaptiveThreshold(dilation, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 5)

          # creating contour
          contours, hierarchy = cv2.findContours(bins.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

          # use convex hull to create polygon around contours
          hull = [cv2.convexHull(c) for c in contours]

          cv2.drawContours(image, hull, -1, (0, 255, 0), 3)

          # line
          # lineypos = 225
          cv2.line(image, (linexpos, lineypos), (linexpos2, lineypos2), (255, 0, 0), 5)
          # lineypos2 = 250
          cv2.line(image, (linexpos, lineypos+25), (linexpos2, lineypos2+25), (0, 255, 0), 5)

          # min area of object
          minarea = 100

          # max area of object
          maxarea = 50000

          # vectors for the x and y locations of contour centroids in current frame
          cxx = np.zeros(len(contours))
          cyy = np.zeros(len(contours))

          for i in range(len(contours)): 

              if hierarchy[0, i, 3] == -1:  # using hierarchy to only count parent contours (contours not within others)

                  area = cv2.contourArea(contours[i])  # area of contour

                  if minarea < area < maxarea:  # area threshold for contour

                      # calculating centroids of contours
                      cnt = contours[i]
                      M = cv2.moments(cnt)
                      cx = int(M['m10'] / M['m00'])
                      cy = int(M['m01'] / M['m00'])

                      if cy > lineypos:  # filters out contours that are above line (y starts at top)

                          # gets bounding points of contour to create rectangle
                          # x,y is top left corner and w,h is width and height
                          x, y, w, h = cv2.boundingRect(cnt)

                          # creates a rectangle around contour
                          cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                          # Prints centroid text in order to double check later on
                          cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                      .3, (0, 0, 255), 1)

                          cv2.drawMarker(image, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1,
                                        line_type=cv2.LINE_AA)

                          # adds centroids that passed previous criteria to centroid list
                          cxx[i] = cx
                          cyy[i] = cy

          # eliminates zero entries (centroids that were not added)
          cxx = cxx[cxx != 0]
          cyy = cyy[cyy != 0]

          # empty list to later check which centroid indices were added to dataframe
          minx_index2 = []
          miny_index2 = []

          # maximum allowable radius for current frame centroid to be considered the same centroid from previous frame
          maxrad = 25

          # The section below keeps track of the centroids and assigns them to old carids or new carids

          if len(cxx):  # if there are centroids in the specified area

              if not carids:  # if carids is empty

                  for i in range(len(cxx)):  # loops through all centroids

                      carids.append(i)  # adds a car id to the empty list carids
                      df[str(carids[i])] = ""  # adds a column to the dataframe corresponding to a carid

                      # assigns the centroid values to the current frame (row) and carid (column)
                      df.at[int(framenumber), str(carids[i])] = [cxx[i], cyy[i]]

                      totalcars = carids[i] + 1  # adds one count to total cars

              else:  # if there are already car ids

                  dx = np.zeros((len(cxx), len(carids)))  # new arrays to calculate deltas
                  dy = np.zeros((len(cyy), len(carids)))  # new arrays to calculate deltas

                  for i in range(len(cxx)):  # loops through all centroids

                      for j in range(len(carids)):  # loops through all recorded car ids

                          # acquires centroid from previous frame for specific carid
                          oldcxcy = df.iloc[int(framenumber - 1)][str(carids[j])]

                          # acquires current frame centroid that doesn't necessarily line up with previous frame centroid
                          curcxcy = np.array([cxx[i], cyy[i]])

                          if not oldcxcy:  # checks if old centroid is empty in case car leaves screen and new car shows

                              continue  # continue to next carid

                          else:  # calculate centroid deltas to compare to current frame position later

                              dx[i, j] = oldcxcy[0] - curcxcy[0]
                              dy[i, j] = oldcxcy[1] - curcxcy[1]

                  for j in range(len(carids)):  # loops through all current car ids

                      sumsum = np.abs(dx[:, j]) + np.abs(dy[:, j])  # sums the deltas wrt to car ids

                      # finds which index carid had the min difference and this is true index
                      correctindextrue = np.argmin(np.abs(sumsum))
                      minx_index = correctindextrue
                      miny_index = correctindextrue

                      # acquires delta values of the minimum deltas in order to check if it is within radius later on
                      mindx = dx[minx_index, j]
                      mindy = dy[miny_index, j]

                      if mindx == 0 and mindy == 0 and np.all(dx[:, j] == 0) and np.all(dy[:, j] == 0):
                          # checks if minimum value is 0 and checks if all deltas are zero since this is empty set
                          # delta could be zero if centroid didn't move

                          continue  # continue to next carid

                      else:

                          # if delta values are less than maximum radius then add that centroid to that specific carid
                          if np.abs(mindx) < maxrad and np.abs(mindy) < maxrad:

                              # adds centroid to corresponding previously existing carid
                              df.at[int(framenumber), str(carids[j])] = [cxx[minx_index], cyy[miny_index]]
                              minx_index2.append(minx_index)  # appends all the indices that were added to previous carids
                              miny_index2.append(miny_index)

                  for i in range(len(cxx)):  # loops through all centroids

                      # if centroid is not in the minindex list then another car needs to be added
                      if i not in minx_index2 and miny_index2:

                          df[str(totalcars)] = ""  # create another column with total cars
                          totalcars = totalcars + 1  # adds another total car the count
                          t = totalcars - 1  # t is a placeholder to total cars
                          carids.append(t)  # append to list of car ids
                          df.at[int(framenumber), str(t)] = [cxx[i], cyy[i]]  # add centroid to the new car id

                      elif curcxcy[0] and not oldcxcy and not minx_index2 and not miny_index2:
                          # checks if current centroid exists but previous centroid does not
                          # new car to be added in case minx_index2 is empty

                          df[str(totalcars)] = ""  # create another column with total cars
                          totalcars = totalcars + 1  # adds another total car the count
                          t = totalcars - 1  # t is a placeholder to total cars
                          carids.append(t)  # append to list of car ids
                          df.at[int(framenumber), str(t)] = [cxx[i], cyy[i]]  # add centroid to the new car id

          # The section below labels the centroids on screen

          currentcars = 0  # current cars on screen
          currentcarsindex = []  # current cars on screen carid index

          for i in range(len(carids)):  # loops through all carids

              if df.at[int(framenumber), str(carids[i])] != '':
                  # checks the current frame to see which car ids are active
                  # by checking in centroid exists on current frame for certain car id

                  currentcars = currentcars + 1  # adds another to current cars on screen
                  currentcarsindex.append(i)  # adds car ids to current cars on screen

          for i in range(currentcars):  # loops through all current car ids on screen

              # grabs centroid of certain carid for current frame
              curcent = df.iloc[int(framenumber)][str(carids[currentcarsindex[i]])]

              # grabs centroid of certain carid for previous frame
              oldcent = df.iloc[int(framenumber - 1)][str(carids[currentcarsindex[i]])]

              if curcent:  # if there is a current centroid

                  # On-screen text for current centroid
                  # cv2.putText(image, "Centroid" + str(curcent[0]) + "," + str(curcent[1]),
                  #             (int(curcent[0]), int(curcent[1])), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

                  cv2.putText(image, "ID:" + str(carids[currentcarsindex[i]]), (int(curcent[0]), int(curcent[1] - 15)),
                              cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

                  cv2.drawMarker(image, (int(curcent[0]), int(curcent[1])), (0, 0, 255), cv2.MARKER_STAR, markerSize=5,
                                thickness=1, line_type=cv2.LINE_AA)

                  if oldcent:  # checks if old centroid exists
                      xstart = oldcent[0] - maxrad
                      ystart = oldcent[1] - maxrad
                      xwidth = oldcent[0] + maxrad
                      yheight = oldcent[1] + maxrad
                      cv2.rectangle(image, (int(xstart), int(ystart)), (int(xwidth), int(yheight)), (0, 125, 0), 1)

                      # checks if old centroid is on or below line and curcent is on or above line
                      # if oldcent[1] >= lineypos2 and curcent[1] <= lineypos2 and carids[
                      #     currentcarsindex[i]] not in caridscrossed:

                      if check(oldcent[0], oldcent[1], eq) and (not check(curcent[0], curcent[1], eq)) and carids[
                          currentcarsindex[i]] not in caridscrossed:

                          carscrossedup += 1
                          cv2.line(image, (0, lineypos2), (width, lineypos2), (0, 0, 255), 5)
                          caridscrossed.append(
                              currentcarsindex[i])  # adds car id to list of count cars to prevent double counting

                      # to count cars and that car hasn't been counted yet
                      # elif oldcent[1] <= lineypos2 and curcent[1] >= lineypos2 and carids[
                          # currentcarsindex[i]] not in caridscrossed:
                      
                      elif (not check(oldcent[0], oldcent[1], eq)) and (check(curcent[0], curcent[1], eq)) and carids[
                          currentcarsindex[i]] not in caridscrossed:

                          carscrosseddown += 1
                          cv2.line(image, (0, lineypos2), (width, lineypos2), (0, 0, 125), 5)
                          caridscrossed.append(currentcarsindex[i])

          # Top left hand corner on-screen text
          cv2.rectangle(image, (0, 0), (250, 100), (255, 0, 0), -1)  # background rectangle for on-screen text

          cv2.putText(image, "Cars in Area: " + str(currentcars), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0), 1)

          cv2.putText(image, "Cars Crossed Up: " + str(carscrossedup), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0),
                      1)

          cv2.putText(image, "Cars Crossed Down: " + str(carscrosseddown), (0, 45), cv2.FONT_HERSHEY_SIMPLEX, .5,
                      (0, 170, 0), 1)

          video.write(image)  # save the current image to video file from earlier

          framenumber += 1

      else:

          break

  print('Task done.. prepared to terminate')

  cap.release()
  return
