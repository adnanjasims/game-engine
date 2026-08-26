#pragma once

#if defined(_WIN32)
#if defined(EOC_BUILDING_SHARED)
#define EOC_API __declspec(dllexport)
#else
#define EOC_API __declspec(dllimport)
#endif
#else
#define EOC_API __attribute__((visibility("default")))
#endif
