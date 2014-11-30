
#ifndef __LOG_H
#define __LOG_H

#include "headers.hpp"
#include "log4cpp/Category.hh"


namespace log4cpp {
	extern Category *log_console;
	extern Category *log_file;

    void initLogs();
}

using log4cpp::log_console;

#endif /* end of include guard: LOG_H */
