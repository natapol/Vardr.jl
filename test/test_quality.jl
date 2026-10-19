using Aqua
using JET

Aqua.test_all(Vardr)
JET.test_package(Vardr; target_defined_modules = true)
