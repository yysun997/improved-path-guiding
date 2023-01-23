if(Qt5_FOUND)
  set(CMAKE_AUTOMOC ON)
  set(CMAKE_AUTORCC ON)
  set(CMAKE_AUTOUIC ON)
  set(CMAKE_INCLUDE_CURRENT_DIR ON)

  if(${COLLADA_DOM_FOUND})
    include_directories(${COLLADA_DOM_INCLUDE_DIRS})
    link_libraries(${COLLADA_DOM_LIBRARIES})
  endif()

  add_mts_executable(mtsgui
    # CPP Sources
    aboutdlg.cpp acknowledgmentdlg.cpp addserverdlg.cpp glwidget.cpp
    importdlg.cpp loaddlg.cpp locateresourcedlg.cpp logwidget.cpp main.cpp
    mainwindow.cpp preview.cpp previewsettingsdlg.cpp programsettingsdlg.cpp
    rendersettingsdlg.cpp save.cpp sceneimporter.cpp sceneinfodlg.cpp
    sceneloader.cpp server.cpp simdtonemap.cpp symlinks_auth.cpp tabbar.cpp
    test_simdtonemap.cpp updatedlg.cpp upgrade.cpp xmltreemodel.cpp

    # Qt5 UI files
    aboutdlg.ui acknowledgmentdlg.ui addserverdlg.ui importdlg.ui loaddlg.ui
    locateresourcedlg.ui mainwindow.ui previewsettingsdlg.ui programsettingsdlg.ui
    rendersettingsdlg.ui sceneinfodlg.ui updatedlg.ui

    # Qt5 Resources
    resources.qrc)

  target_link_libraries(mtsgui
    mitsuba-converter
    mitsuba-core
    mitsuba-hw
    mitsuba-render
    ${GL_LIBRARIES}
    ${Qt5_LIBRARIES})
endif()
