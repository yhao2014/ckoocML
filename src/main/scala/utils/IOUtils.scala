package utils

import java.io.File

/**
  * io工具类
  *
  * Created by yhao on 2017/3/9.
  */
object IOUtils {

  /**
    * 删除指定文件或目录及其子目录和文件
    *
    * @param file 待删除文件/目录路径
    * @return 是否已删除
    */
  def delDir(file: File): Boolean = {
    if (file.isDirectory) {
      val subFileList = file.listFiles()
      for (subFile <- subFileList) {
        delDir(subFile)
      }
      file.delete()
    } else {
      file.delete()
    }
  }
}
