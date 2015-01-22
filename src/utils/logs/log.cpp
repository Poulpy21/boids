

#include "log.hpp"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>

#include "log4cpp/Category.hh"
#include "log4cpp/Appender.hh"
#include "log4cpp/FileAppender.hh"
#include "log4cpp/OstreamAppender.hh"
#include "log4cpp/Layout.hh"
#include "log4cpp/BasicLayout.hh"
#include "log4cpp/Priority.hh"
#include "log4cpp/PatternLayout.hh"

#include "defines.hpp"

namespace log4cpp {

    Category *log_console = &Category::getRoot();

    void initLogs() {

        log_console->setPriority(Priority::CONSOLE_LOG_LEVEL);

        log4cpp::PatternLayout *layout = new log4cpp::PatternLayout();
        log4cpp::PatternLayout *layout2 = new log4cpp::PatternLayout();
        layout->setConversionPattern("%d{%H:%M:%S} %p %c %x: %m%n");
        layout2->setConversionPattern("%d{%H:%M:%S} %p %c %x: %m%n");

        log4cpp::Appender *appender_console = new log4cpp::OstreamAppender("console", &std::cout);
        appender_console->setLayout(layout);
        appender_console->setThreshold(Priority::DEBUG);
        log_console->addAppender(appender_console);

        log4cpp::Appender *appender_file = new log4cpp::FileAppender("default", "program.log");
        appender_file->setLayout(layout2);
        appender_file->setThreshold(Priority::DEBUG);
        log_console->addAppender(appender_file);
    }
}

