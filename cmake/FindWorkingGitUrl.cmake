# find_working_git_url(<out_var> URLS <url1> [<url2> ...])
#
# Probes each URL in order with `git ls-remote` and sets <out_var> to the
# first one that succeeds. Suppresses git credential prompts so failures
# return quickly instead of hanging.
function(find_working_git_url out_var)
  cmake_parse_arguments(ARG "" "" "URLS" ${ARGN})

  set(ENV{GIT_TERMINAL_PROMPT} 0)

  foreach(url IN LISTS ARG_URLS)
    message(STATUS "find_working_git_url: probing ${url}")
    execute_process(
      COMMAND git ls-remote --exit-code "${url}" HEAD
      RESULT_VARIABLE result
      OUTPUT_QUIET
      ERROR_QUIET
      TIMEOUT 10
    )
    if(result EQUAL 0)
      message(STATUS "find_working_git_url: using ${url}")
      set(${out_var} "${url}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
endfunction()