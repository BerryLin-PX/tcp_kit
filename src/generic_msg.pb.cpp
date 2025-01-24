// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: generic_msg.proto

#include "include/network/generic_msg.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>

PROTOBUF_PRAGMA_INIT_SEG

namespace _pb = ::PROTOBUF_NAMESPACE_ID;
namespace _pbi = _pb::internal;

namespace tcp_kit {
PROTOBUF_CONSTEXPR BasicType::BasicType(
    ::_pbi::ConstantInitialized)
  : _oneof_case_{}{}
struct BasicTypeDefaultTypeInternal {
  PROTOBUF_CONSTEXPR BasicTypeDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~BasicTypeDefaultTypeInternal() {}
  union {
    BasicType _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 BasicTypeDefaultTypeInternal _BasicType_default_instance_;
PROTOBUF_CONSTEXPR GenericMsg::GenericMsg(
    ::_pbi::ConstantInitialized)
  : params_()
  , api_(&::_pbi::fixed_address_empty_string, ::_pbi::ConstantInitialized{})
  , body_(nullptr){}
struct GenericMsgDefaultTypeInternal {
  PROTOBUF_CONSTEXPR GenericMsgDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~GenericMsgDefaultTypeInternal() {}
  union {
    GenericMsg _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 GenericMsgDefaultTypeInternal _GenericMsg_default_instance_;
}  // namespace tcp_kit
static ::_pb::Metadata file_level_metadata_generic_5fmsg_2eproto[2];
static constexpr ::_pb::EnumDescriptor const** file_level_enum_descriptors_generic_5fmsg_2eproto = nullptr;
static constexpr ::_pb::ServiceDescriptor const** file_level_service_descriptors_generic_5fmsg_2eproto = nullptr;

const uint32_t TableStruct_generic_5fmsg_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::tcp_kit::BasicType, _internal_metadata_),
  ~0u,  // no _extensions_
  PROTOBUF_FIELD_OFFSET(::tcp_kit::BasicType, _oneof_case_[0]),
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  ::_pbi::kInvalidFieldOffsetTag,
  ::_pbi::kInvalidFieldOffsetTag,
  ::_pbi::kInvalidFieldOffsetTag,
  ::_pbi::kInvalidFieldOffsetTag,
  ::_pbi::kInvalidFieldOffsetTag,
  ::_pbi::kInvalidFieldOffsetTag,
  ::_pbi::kInvalidFieldOffsetTag,
  ::_pbi::kInvalidFieldOffsetTag,
  PROTOBUF_FIELD_OFFSET(::tcp_kit::BasicType, value_),
  PROTOBUF_FIELD_OFFSET(::tcp_kit::GenericMsg, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::tcp_kit::GenericMsg, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::tcp_kit::GenericMsg, api_),
  PROTOBUF_FIELD_OFFSET(::tcp_kit::GenericMsg, params_),
  PROTOBUF_FIELD_OFFSET(::tcp_kit::GenericMsg, body_),
  ~0u,
  ~0u,
  0,
};
static const ::_pbi::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, -1, sizeof(::tcp_kit::BasicType)},
  { 15, 24, -1, sizeof(::tcp_kit::GenericMsg)},
};

static const ::_pb::Message* const file_default_instances[] = {
  &::tcp_kit::_BasicType_default_instance_._instance,
  &::tcp_kit::_GenericMsg_default_instance_._instance,
};

const char descriptor_table_protodef_generic_5fmsg_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\021generic_msg.proto\022\007tcp_kit\032\031google/pro"
  "tobuf/any.proto\"\206\001\n\tBasicType\022\r\n\003u32\030\001 \001"
  "(\rH\000\022\r\n\003s32\030\002 \001(\005H\000\022\r\n\003u64\030\003 \001(\004H\000\022\r\n\003s6"
  "4\030\004 \001(\003H\000\022\013\n\001f\030\005 \001(\002H\000\022\013\n\001d\030\006 \001(\001H\000\022\013\n\001b"
  "\030\007 \001(\010H\000\022\r\n\003str\030\010 \001(\tH\000B\007\n\005value\"o\n\nGene"
  "ricMsg\022\013\n\003api\030\001 \001(\t\022\"\n\006params\030\002 \003(\0132\022.tc"
  "p_kit.BasicType\022\'\n\004body\030\003 \001(\0132\024.google.p"
  "rotobuf.AnyH\000\210\001\001B\007\n\005_bodyb\006proto3"
  ;
static const ::_pbi::DescriptorTable* const descriptor_table_generic_5fmsg_2eproto_deps[1] = {
  &::descriptor_table_google_2fprotobuf_2fany_2eproto,
};
static ::_pbi::once_flag descriptor_table_generic_5fmsg_2eproto_once;
const ::_pbi::DescriptorTable descriptor_table_generic_5fmsg_2eproto = {
    false, false, 313, descriptor_table_protodef_generic_5fmsg_2eproto,
    "generic_msg.proto",
    &descriptor_table_generic_5fmsg_2eproto_once, descriptor_table_generic_5fmsg_2eproto_deps, 1, 2,
    schemas, file_default_instances, TableStruct_generic_5fmsg_2eproto::offsets,
    file_level_metadata_generic_5fmsg_2eproto, file_level_enum_descriptors_generic_5fmsg_2eproto,
    file_level_service_descriptors_generic_5fmsg_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::_pbi::DescriptorTable* descriptor_table_generic_5fmsg_2eproto_getter() {
  return &descriptor_table_generic_5fmsg_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY2 static ::_pbi::AddDescriptorsRunner dynamic_init_dummy_generic_5fmsg_2eproto(&descriptor_table_generic_5fmsg_2eproto);
namespace tcp_kit {

// ===================================================================

class BasicType::_Internal {
 public:
};

BasicType::BasicType(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor();
  // @@protoc_insertion_point(arena_constructor:tcp_kit.BasicType)
}
BasicType::BasicType(const BasicType& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  clear_has_value();
  switch (from.value_case()) {
    case kU32: {
      _internal_set_u32(from._internal_u32());
      break;
    }
    case kS32: {
      _internal_set_s32(from._internal_s32());
      break;
    }
    case kU64: {
      _internal_set_u64(from._internal_u64());
      break;
    }
    case kS64: {
      _internal_set_s64(from._internal_s64());
      break;
    }
    case kF: {
      _internal_set_f(from._internal_f());
      break;
    }
    case kD: {
      _internal_set_d(from._internal_d());
      break;
    }
    case kB: {
      _internal_set_b(from._internal_b());
      break;
    }
    case kStr: {
      _internal_set_str(from._internal_str());
      break;
    }
    case VALUE_NOT_SET: {
      break;
    }
  }
  // @@protoc_insertion_point(copy_constructor:tcp_kit.BasicType)
}

inline void BasicType::SharedCtor() {
clear_has_value();
}

BasicType::~BasicType() {
  // @@protoc_insertion_point(destructor:tcp_kit.BasicType)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void BasicType::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  if (has_value()) {
    clear_value();
  }
}

void BasicType::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void BasicType::clear_value() {
// @@protoc_insertion_point(one_of_clear_start:tcp_kit.BasicType)
  switch (value_case()) {
    case kU32: {
      // No need to clear
      break;
    }
    case kS32: {
      // No need to clear
      break;
    }
    case kU64: {
      // No need to clear
      break;
    }
    case kS64: {
      // No need to clear
      break;
    }
    case kF: {
      // No need to clear
      break;
    }
    case kD: {
      // No need to clear
      break;
    }
    case kB: {
      // No need to clear
      break;
    }
    case kStr: {
      value_.str_.Destroy();
      break;
    }
    case VALUE_NOT_SET: {
      break;
    }
  }
  _oneof_case_[0] = VALUE_NOT_SET;
}


void BasicType::Clear() {
// @@protoc_insertion_point(message_clear_start:tcp_kit.BasicType)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  clear_value();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* BasicType::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // uint32 u32 = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 8)) {
          _internal_set_u32(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr));
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // int32 s32 = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 16)) {
          _internal_set_s32(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr));
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // uint64 u64 = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 24)) {
          _internal_set_u64(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr));
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // int64 s64 = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 32)) {
          _internal_set_s64(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr));
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // float f = 5;
      case 5:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 45)) {
          _internal_set_f(::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr));
          ptr += sizeof(float);
        } else
          goto handle_unusual;
        continue;
      // double d = 6;
      case 6:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 49)) {
          _internal_set_d(::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<double>(ptr));
          ptr += sizeof(double);
        } else
          goto handle_unusual;
        continue;
      // bool b = 7;
      case 7:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 56)) {
          _internal_set_b(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr));
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // string str = 8;
      case 8:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 66)) {
          auto str = _internal_mutable_str();
          ptr = ::_pbi::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(ptr);
          CHK_(::_pbi::VerifyUTF8(str, "tcp_kit.BasicType.str"));
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

uint8_t* BasicType::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:tcp_kit.BasicType)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // uint32 u32 = 1;
  if (_internal_has_u32()) {
    target = stream->EnsureSpace(target);
    target = ::_pbi::WireFormatLite::WriteUInt32ToArray(1, this->_internal_u32(), target);
  }

  // int32 s32 = 2;
  if (_internal_has_s32()) {
    target = stream->EnsureSpace(target);
    target = ::_pbi::WireFormatLite::WriteInt32ToArray(2, this->_internal_s32(), target);
  }

  // uint64 u64 = 3;
  if (_internal_has_u64()) {
    target = stream->EnsureSpace(target);
    target = ::_pbi::WireFormatLite::WriteUInt64ToArray(3, this->_internal_u64(), target);
  }

  // int64 s64 = 4;
  if (_internal_has_s64()) {
    target = stream->EnsureSpace(target);
    target = ::_pbi::WireFormatLite::WriteInt64ToArray(4, this->_internal_s64(), target);
  }

  // float f = 5;
  if (_internal_has_f()) {
    target = stream->EnsureSpace(target);
    target = ::_pbi::WireFormatLite::WriteFloatToArray(5, this->_internal_f(), target);
  }

  // double d = 6;
  if (_internal_has_d()) {
    target = stream->EnsureSpace(target);
    target = ::_pbi::WireFormatLite::WriteDoubleToArray(6, this->_internal_d(), target);
  }

  // bool b = 7;
  if (_internal_has_b()) {
    target = stream->EnsureSpace(target);
    target = ::_pbi::WireFormatLite::WriteBoolToArray(7, this->_internal_b(), target);
  }

  // string str = 8;
  if (_internal_has_str()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_str().data(), static_cast<int>(this->_internal_str().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tcp_kit.BasicType.str");
    target = stream->WriteStringMaybeAliased(
        8, this->_internal_str(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tcp_kit.BasicType)
  return target;
}

size_t BasicType::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tcp_kit.BasicType)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  switch (value_case()) {
    // uint32 u32 = 1;
    case kU32: {
      total_size += ::_pbi::WireFormatLite::UInt32SizePlusOne(this->_internal_u32());
      break;
    }
    // int32 s32 = 2;
    case kS32: {
      total_size += ::_pbi::WireFormatLite::Int32SizePlusOne(this->_internal_s32());
      break;
    }
    // uint64 u64 = 3;
    case kU64: {
      total_size += ::_pbi::WireFormatLite::UInt64SizePlusOne(this->_internal_u64());
      break;
    }
    // int64 s64 = 4;
    case kS64: {
      total_size += ::_pbi::WireFormatLite::Int64SizePlusOne(this->_internal_s64());
      break;
    }
    // float f = 5;
    case kF: {
      total_size += 1 + 4;
      break;
    }
    // double d = 6;
    case kD: {
      total_size += 1 + 8;
      break;
    }
    // bool b = 7;
    case kB: {
      total_size += 1 + 1;
      break;
    }
    // string str = 8;
    case kStr: {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
          this->_internal_str());
      break;
    }
    case VALUE_NOT_SET: {
      break;
    }
  }
  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData BasicType::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    BasicType::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*BasicType::GetClassData() const { return &_class_data_; }

void BasicType::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<BasicType *>(to)->MergeFrom(
      static_cast<const BasicType &>(from));
}


void BasicType::MergeFrom(const BasicType& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:tcp_kit.BasicType)
  GOOGLE_DCHECK_NE(&from, this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  switch (from.value_case()) {
    case kU32: {
      _internal_set_u32(from._internal_u32());
      break;
    }
    case kS32: {
      _internal_set_s32(from._internal_s32());
      break;
    }
    case kU64: {
      _internal_set_u64(from._internal_u64());
      break;
    }
    case kS64: {
      _internal_set_s64(from._internal_s64());
      break;
    }
    case kF: {
      _internal_set_f(from._internal_f());
      break;
    }
    case kD: {
      _internal_set_d(from._internal_d());
      break;
    }
    case kB: {
      _internal_set_b(from._internal_b());
      break;
    }
    case kStr: {
      _internal_set_str(from._internal_str());
      break;
    }
    case VALUE_NOT_SET: {
      break;
    }
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void BasicType::CopyFrom(const BasicType& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tcp_kit.BasicType)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool BasicType::IsInitialized() const {
  return true;
}

void BasicType::InternalSwap(BasicType* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(value_, other->value_);
  swap(_oneof_case_[0], other->_oneof_case_[0]);
}

::PROTOBUF_NAMESPACE_ID::Metadata BasicType::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_generic_5fmsg_2eproto_getter, &descriptor_table_generic_5fmsg_2eproto_once,
      file_level_metadata_generic_5fmsg_2eproto[0]);
}

// ===================================================================

class GenericMsg::_Internal {
 public:
  using HasBits = decltype(std::declval<GenericMsg>()._has_bits_);
  static const ::PROTOBUF_NAMESPACE_ID::Any& body(const GenericMsg* msg);
  static void set_has_body(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
};

const ::PROTOBUF_NAMESPACE_ID::Any&
GenericMsg::_Internal::body(const GenericMsg* msg) {
  return *msg->body_;
}
void GenericMsg::clear_body() {
  if (body_ != nullptr) body_->Clear();
  _has_bits_[0] &= ~0x00000001u;
}
GenericMsg::GenericMsg(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned),
  params_(arena) {
  SharedCtor();
  // @@protoc_insertion_point(arena_constructor:tcp_kit.GenericMsg)
}
GenericMsg::GenericMsg(const GenericMsg& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_),
      params_(from.params_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  api_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    api_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_api().empty()) {
    api_.Set(from._internal_api(), 
      GetArenaForAllocation());
  }
  if (from._internal_has_body()) {
    body_ = new ::PROTOBUF_NAMESPACE_ID::Any(*from.body_);
  } else {
    body_ = nullptr;
  }
  // @@protoc_insertion_point(copy_constructor:tcp_kit.GenericMsg)
}

inline void GenericMsg::SharedCtor() {
api_.InitDefault();
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  api_.Set("", GetArenaForAllocation());
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
body_ = nullptr;
}

GenericMsg::~GenericMsg() {
  // @@protoc_insertion_point(destructor:tcp_kit.GenericMsg)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void GenericMsg::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  api_.Destroy();
  if (this != internal_default_instance()) delete body_;
}

void GenericMsg::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void GenericMsg::Clear() {
// @@protoc_insertion_point(message_clear_start:tcp_kit.GenericMsg)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  params_.Clear();
  api_.ClearToEmpty();
  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000001u) {
    GOOGLE_DCHECK(body_ != nullptr);
    body_->Clear();
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* GenericMsg::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // string api = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          auto str = _internal_mutable_api();
          ptr = ::_pbi::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(ptr);
          CHK_(::_pbi::VerifyUTF8(str, "tcp_kit.GenericMsg.api"));
        } else
          goto handle_unusual;
        continue;
      // repeated .tcp_kit.BasicType params = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 18)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(_internal_add_params(), ptr);
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<18>(ptr));
        } else
          goto handle_unusual;
        continue;
      // optional .google.protobuf.Any body = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 26)) {
          ptr = ctx->ParseMessage(_internal_mutable_body(), ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  _has_bits_.Or(has_bits);
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

uint8_t* GenericMsg::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:tcp_kit.GenericMsg)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // string api = 1;
  if (!this->_internal_api().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_api().data(), static_cast<int>(this->_internal_api().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tcp_kit.GenericMsg.api");
    target = stream->WriteStringMaybeAliased(
        1, this->_internal_api(), target);
  }

  // repeated .tcp_kit.BasicType params = 2;
  for (unsigned i = 0,
      n = static_cast<unsigned>(this->_internal_params_size()); i < n; i++) {
    const auto& repfield = this->_internal_params(i);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
        InternalWriteMessage(2, repfield, repfield.GetCachedSize(), target, stream);
  }

  // optional .google.protobuf.Any body = 3;
  if (_internal_has_body()) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(3, _Internal::body(this),
        _Internal::body(this).GetCachedSize(), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tcp_kit.GenericMsg)
  return target;
}

size_t GenericMsg::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tcp_kit.GenericMsg)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated .tcp_kit.BasicType params = 2;
  total_size += 1UL * this->_internal_params_size();
  for (const auto& msg : this->params_) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(msg);
  }

  // string api = 1;
  if (!this->_internal_api().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_api());
  }

  // optional .google.protobuf.Any body = 3;
  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000001u) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
        *body_);
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData GenericMsg::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    GenericMsg::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GenericMsg::GetClassData() const { return &_class_data_; }

void GenericMsg::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<GenericMsg *>(to)->MergeFrom(
      static_cast<const GenericMsg &>(from));
}


void GenericMsg::MergeFrom(const GenericMsg& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:tcp_kit.GenericMsg)
  GOOGLE_DCHECK_NE(&from, this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  params_.MergeFrom(from.params_);
  if (!from._internal_api().empty()) {
    _internal_set_api(from._internal_api());
  }
  if (from._internal_has_body()) {
    _internal_mutable_body()->::PROTOBUF_NAMESPACE_ID::Any::MergeFrom(from._internal_body());
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void GenericMsg::CopyFrom(const GenericMsg& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tcp_kit.GenericMsg)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool GenericMsg::IsInitialized() const {
  return true;
}

void GenericMsg::InternalSwap(GenericMsg* other) {
  using std::swap;
  auto* lhs_arena = GetArenaForAllocation();
  auto* rhs_arena = other->GetArenaForAllocation();
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  params_.InternalSwap(&other->params_);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &api_, lhs_arena,
      &other->api_, rhs_arena
  );
  swap(body_, other->body_);
}

::PROTOBUF_NAMESPACE_ID::Metadata GenericMsg::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_generic_5fmsg_2eproto_getter, &descriptor_table_generic_5fmsg_2eproto_once,
      file_level_metadata_generic_5fmsg_2eproto[1]);
}

// @@protoc_insertion_point(namespace_scope)
}  // namespace tcp_kit
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::tcp_kit::BasicType*
Arena::CreateMaybeMessage< ::tcp_kit::BasicType >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tcp_kit::BasicType >(arena);
}
template<> PROTOBUF_NOINLINE ::tcp_kit::GenericMsg*
Arena::CreateMaybeMessage< ::tcp_kit::GenericMsg >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tcp_kit::GenericMsg >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
