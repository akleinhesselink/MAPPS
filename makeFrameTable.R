####
#### 
#### 

rm(list = ls() )

getCreationTime = function( frameInfo , creationTimeHeader, patternDateTime ) { 
  
  ##### scan through the frameInfo file 
  ##### return the creation time and date
  
  creationTimeMatch = grep(creationTimeHeader, frameInfo)[1]
  creationTimeLine = frameInfo[creationTimeMatch]
  
  startTimeMatch = regexec ( pattern= patternDateTime, text= creationTimeLine )
  startTime = as.POSIXct( regmatches( creationTimeLine, startTimeMatch )[[1]][1])
  
  return(startTime)
}

getFrameLines = function( frameInfo , headerPattern ){ 

  #### find lines with info on frames 
  #### return lines as list 
  
  linesMatches = grep(pattern= headerPattern, x=frameInfo )
  infoLines = as.list(frameInfo[linesMatches ])
  return(infoLines)  
}


getFrameNo = function( infoLine, patternFrameNo ){ 
  
  ##### reads line and returns frame number  
  
  frameNo.match = regexec ( patternFrameNo,  infoLine )
  frameNo = as.numeric(regmatches( infoLine, m = frameNo.match)[[1]][2] )
  return(frameNo)  
}

getFrameTime = function( infoLine, patternTimeSecs ){ 
  
  #### Get the time in seconds for a still frame 
  
  timeSecs.match = regexec ( patternTimeSecs,  infoLine )
  timeSecs = as.numeric( regmatches( infoLine, m=timeSecs.match )[[1]][2] )
  return(timeSecs)  
}


frameInfo = scan( '~/Documents/Kleinhesselink/autoDigitizing/stills.txt' , 'text', sep = '\n')

CT = 'creation_time[ ]+:'
pattern_DT = '([0-9]+-[0-9]+-[0-9]+[ ]+[0-9]+:[0-9]+:[0-9]+)'

startTime = getCreationTime( frameInfo= frameInfo, creationTimeHeader=CT, patternDateTime=pattern_DT)
startTime

frameHeader = 'Parsed_showinfo_.*\\][ ]+n:'
frameLines = getFrameLines( frameInfo , headerPattern = frameHeader )

patternFrameNo = 'n:[ ]+([0-9]+)[ ]+pts:'
frameNo = getFrameNo( infoLine= frameLines[[2]], patternFrameNo = patternFrameNo)

patternFrameTime = 'pts_time:([0-9]+(\\.?[0-9]+)?)[ ]+pos'
frameTime = getFrameTime(infoLine = frameLines[[1]], patternTimeSecs=patternFrameTime  )

frameTable = data.frame( frame = 1 + as.numeric(lapply( frameLines, FUN = 'getFrameNo',  patternFrameNo )), timeSecs = NA)
frameTable$timeSecs = as.numeric ( lapply( frameLines, FUN = 'getFrameTime', patternFrameTime ))

frameTable$dateTime = startTime + frameTable$timeSecs


frameTable