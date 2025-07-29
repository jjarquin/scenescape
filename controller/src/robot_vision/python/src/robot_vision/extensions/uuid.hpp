// SPDX-FileCopyrightText: 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/nil_generator.hpp>
#include <boost/uuid/string_generator.hpp>

namespace PYBIND11_NAMESPACE { namespace detail {
    template <> struct type_caster<boost::uuids::uuid> {
    public:
        /**
         * This macro establishes the name 'uuid' in
         * function signatures and declares a local variable
         * 'value' of type uuid
         */
        PYBIND11_TYPE_CASTER(boost::uuids::uuid, const_name("uuid"));

        /**
         * Python->C++ : convert a PyObject to uuid
         */
        bool load(handle src, bool) {
            /* Extract PyObject from handle */
            PyObject *source = src.ptr();

            if (source == Py_None)
            {
              value = boost::uuids::nil_uuid();
              return true;
            }
            if (PyUnicode_Check(source))
            {
              std::string uuid_string(PyUnicode_AsUTF8(source));
              try
              {
                value = string_generator(uuid_string);
              }
              catch (const std::runtime_error &e)
              {
                return false;
              }

              return true;
            }
            else
            {
              return false;
            }
        }

        /**
         * C++ -> Python): convert an uuid instance into
         * a Python object.
         */
        static handle cast(boost::uuids::uuid src, return_value_policy /* policy */, handle /* parent */) {
            std::string uuid_string(boost::uuids::to_string(src));
            return PyUnicode_FromString(uuid_string.c_str());
        }
        boost::uuids::string_generator string_generator;
    };
}} // namespace PYBIND11_NAMESPACE::detail