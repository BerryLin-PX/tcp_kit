// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: generic_msg.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_generic_5fmsg_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_generic_5fmsg_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3020000
#error_flag This file was generated by a newer version of protoc which is
#error_flag incompatible with your Protocol Buffer headers. Please update
#error_flag your headers.
#endif
#if 3020003 < PROTOBUF_MIN_PROTOC_VERSION
#error_flag This file was generated by an older version of protoc which is
#error_flag incompatible with your Protocol Buffer headers. Please
#error_flag regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
#include <google/protobuf/any.pb.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_generic_5fmsg_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_generic_5fmsg_2eproto {
  static const uint32_t offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_generic_5fmsg_2eproto;
namespace tcp_kit {
class BasicType;
struct BasicTypeDefaultTypeInternal;
extern BasicTypeDefaultTypeInternal _BasicType_default_instance_;
class GenericMsg;
struct GenericMsgDefaultTypeInternal;
extern GenericMsgDefaultTypeInternal _GenericMsg_default_instance_;
}  // namespace tcp_kit
PROTOBUF_NAMESPACE_OPEN
template<> ::tcp_kit::BasicType* Arena::CreateMaybeMessage<::tcp_kit::BasicType>(Arena*);
template<> ::tcp_kit::GenericMsg* Arena::CreateMaybeMessage<::tcp_kit::GenericMsg>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace tcp_kit {

// ===================================================================

class BasicType final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:tcp_kit.BasicType) */ {
 public:
  inline BasicType() : BasicType(nullptr) {}
  ~BasicType() override;
  explicit PROTOBUF_CONSTEXPR BasicType(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  BasicType(const BasicType& from);
  BasicType(BasicType&& from) noexcept
    : BasicType() {
    *this = ::std::move(from);
  }

  inline BasicType& operator=(const BasicType& from) {
    CopyFrom(from);
    return *this;
  }
  inline BasicType& operator=(BasicType&& from) noexcept {
    if (this == &from) return *this;
    if (GetOwningArena() == from.GetOwningArena()
  #ifdef PROTOBUF_FORCE_COPY_IN_MOVE
        && GetOwningArena() != nullptr
  #endif  // !PROTOBUF_FORCE_COPY_IN_MOVE
    ) {
      InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return default_instance().GetMetadata().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return default_instance().GetMetadata().reflection;
  }
  static const BasicType& default_instance() {
    return *internal_default_instance();
  }
  enum ValueCase {
    kU32 = 1,
    kS32 = 2,
    kU64 = 3,
    kS64 = 4,
    kF = 5,
    kD = 6,
    kB = 7,
    kStr = 8,
    VALUE_NOT_SET = 0,
  };

  static inline const BasicType* internal_default_instance() {
    return reinterpret_cast<const BasicType*>(
               &_BasicType_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(BasicType& a, BasicType& b) {
    a.Swap(&b);
  }
  inline void Swap(BasicType* other) {
    if (other == this) return;
  #ifdef PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() != nullptr &&
        GetOwningArena() == other->GetOwningArena()) {
   #else  // PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() == other->GetOwningArena()) {
  #endif  // !PROTOBUF_FORCE_COPY_IN_SWAP
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(BasicType* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  BasicType* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<BasicType>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const BasicType& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom(const BasicType& from);
  private:
  static void MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to, const ::PROTOBUF_NAMESPACE_ID::Message& from);
  public:
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  uint8_t* _InternalSerialize(
      uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(BasicType* other);

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "tcp_kit.BasicType";
  }
  protected:
  explicit BasicType(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kU32FieldNumber = 1,
    kS32FieldNumber = 2,
    kU64FieldNumber = 3,
    kS64FieldNumber = 4,
    kFFieldNumber = 5,
    kDFieldNumber = 6,
    kBFieldNumber = 7,
    kStrFieldNumber = 8,
  };
  // uint32 u32 = 1;
  bool has_u32() const;
  private:
  bool _internal_has_u32() const;
  public:
  void clear_u32();
  uint32_t u32() const;
  void set_u32(uint32_t value);
  private:
  uint32_t _internal_u32() const;
  void _internal_set_u32(uint32_t value);
  public:

  // int32 s32 = 2;
  bool has_s32() const;
  private:
  bool _internal_has_s32() const;
  public:
  void clear_s32();
  int32_t s32() const;
  void set_s32(int32_t value);
  private:
  int32_t _internal_s32() const;
  void _internal_set_s32(int32_t value);
  public:

  // uint64 u64 = 3;
  bool has_u64() const;
  private:
  bool _internal_has_u64() const;
  public:
  void clear_u64();
  uint64_t u64() const;
  void set_u64(uint64_t value);
  private:
  uint64_t _internal_u64() const;
  void _internal_set_u64(uint64_t value);
  public:

  // int64 s64 = 4;
  bool has_s64() const;
  private:
  bool _internal_has_s64() const;
  public:
  void clear_s64();
  int64_t s64() const;
  void set_s64(int64_t value);
  private:
  int64_t _internal_s64() const;
  void _internal_set_s64(int64_t value);
  public:

  // float f = 5;
  bool has_f() const;
  private:
  bool _internal_has_f() const;
  public:
  void clear_f();
  float f() const;
  void set_f(float value);
  private:
  float _internal_f() const;
  void _internal_set_f(float value);
  public:

  // double d = 6;
  bool has_d() const;
  private:
  bool _internal_has_d() const;
  public:
  void clear_d();
  double d() const;
  void set_d(double value);
  private:
  double _internal_d() const;
  void _internal_set_d(double value);
  public:

  // bool b = 7;
  bool has_b() const;
  private:
  bool _internal_has_b() const;
  public:
  void clear_b();
  bool b() const;
  void set_b(bool value);
  private:
  bool _internal_b() const;
  void _internal_set_b(bool value);
  public:

  // string str = 8;
  bool has_str() const;
  private:
  bool _internal_has_str() const;
  public:
  void clear_str();
  const std::string& str() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_str(ArgT0&& arg0, ArgT... args);
  std::string* mutable_str();
  PROTOBUF_NODISCARD std::string* release_str();
  void set_allocated_str(std::string* str);
  private:
  const std::string& _internal_str() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_str(const std::string& value);
  std::string* _internal_mutable_str();
  public:

  void clear_value();
  ValueCase value_case() const;
  // @@protoc_insertion_point(class_scope:tcp_kit.BasicType)
 private:
  class _Internal;
  void set_has_u32();
  void set_has_s32();
  void set_has_u64();
  void set_has_s64();
  void set_has_f();
  void set_has_d();
  void set_has_b();
  void set_has_str();

  inline bool has_value() const;
  inline void clear_has_value();

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  union ValueUnion {
    constexpr ValueUnion() : _constinit_{} {}
      ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized _constinit_;
    uint32_t u32_;
    int32_t s32_;
    uint64_t u64_;
    int64_t s64_;
    float f_;
    double d_;
    bool b_;
    ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr str_;
  } value_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  uint32_t _oneof_case_[1];

  friend struct ::TableStruct_generic_5fmsg_2eproto;
};
// -------------------------------------------------------------------

class GenericMsg final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:tcp_kit.GenericMsg) */ {
 public:
  inline GenericMsg() : GenericMsg(nullptr) {}
  ~GenericMsg() override;
  explicit PROTOBUF_CONSTEXPR GenericMsg(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  GenericMsg(const GenericMsg& from);
  GenericMsg(GenericMsg&& from) noexcept
    : GenericMsg() {
    *this = ::std::move(from);
  }

  inline GenericMsg& operator=(const GenericMsg& from) {
    CopyFrom(from);
    return *this;
  }
  inline GenericMsg& operator=(GenericMsg&& from) noexcept {
    if (this == &from) return *this;
    if (GetOwningArena() == from.GetOwningArena()
  #ifdef PROTOBUF_FORCE_COPY_IN_MOVE
        && GetOwningArena() != nullptr
  #endif  // !PROTOBUF_FORCE_COPY_IN_MOVE
    ) {
      InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return default_instance().GetMetadata().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return default_instance().GetMetadata().reflection;
  }
  static const GenericMsg& default_instance() {
    return *internal_default_instance();
  }
  static inline const GenericMsg* internal_default_instance() {
    return reinterpret_cast<const GenericMsg*>(
               &_GenericMsg_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(GenericMsg& a, GenericMsg& b) {
    a.Swap(&b);
  }
  inline void Swap(GenericMsg* other) {
    if (other == this) return;
  #ifdef PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() != nullptr &&
        GetOwningArena() == other->GetOwningArena()) {
   #else  // PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() == other->GetOwningArena()) {
  #endif  // !PROTOBUF_FORCE_COPY_IN_SWAP
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(GenericMsg* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  GenericMsg* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<GenericMsg>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const GenericMsg& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom(const GenericMsg& from);
  private:
  static void MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to, const ::PROTOBUF_NAMESPACE_ID::Message& from);
  public:
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  uint8_t* _InternalSerialize(
      uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(GenericMsg* other);

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "tcp_kit.GenericMsg";
  }
  protected:
  explicit GenericMsg(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kParamsFieldNumber = 2,
    kApiFieldNumber = 1,
    kBodyFieldNumber = 3,
  };
  // repeated .tcp_kit.BasicType params = 2;
  int params_size() const;
  private:
  int _internal_params_size() const;
  public:
  void clear_params();
  ::tcp_kit::BasicType* mutable_params(int index);
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::tcp_kit::BasicType >*
      mutable_params();
  private:
  const ::tcp_kit::BasicType& _internal_params(int index) const;
  ::tcp_kit::BasicType* _internal_add_params();
  public:
  const ::tcp_kit::BasicType& params(int index) const;
  ::tcp_kit::BasicType* add_params();
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::tcp_kit::BasicType >&
      params() const;

  // string api = 1;
  void clear_api();
  const std::string& api() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_api(ArgT0&& arg0, ArgT... args);
  std::string* mutable_api();
  PROTOBUF_NODISCARD std::string* release_api();
  void set_allocated_api(std::string* api);
  private:
  const std::string& _internal_api() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_api(const std::string& value);
  std::string* _internal_mutable_api();
  public:

  // optional .google.protobuf.Any body = 3;
  bool has_body() const;
  private:
  bool _internal_has_body() const;
  public:
  void clear_body();
  const ::PROTOBUF_NAMESPACE_ID::Any& body() const;
  PROTOBUF_NODISCARD ::PROTOBUF_NAMESPACE_ID::Any* release_body();
  ::PROTOBUF_NAMESPACE_ID::Any* mutable_body();
  void set_allocated_body(::PROTOBUF_NAMESPACE_ID::Any* body);
  private:
  const ::PROTOBUF_NAMESPACE_ID::Any& _internal_body() const;
  ::PROTOBUF_NAMESPACE_ID::Any* _internal_mutable_body();
  public:
  void unsafe_arena_set_allocated_body(
      ::PROTOBUF_NAMESPACE_ID::Any* body);
  ::PROTOBUF_NAMESPACE_ID::Any* unsafe_arena_release_body();

  // @@protoc_insertion_point(class_scope:tcp_kit.GenericMsg)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::tcp_kit::BasicType > params_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr api_;
  ::PROTOBUF_NAMESPACE_ID::Any* body_;
  friend struct ::TableStruct_generic_5fmsg_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// BasicType

// uint32 u32 = 1;
inline bool BasicType::_internal_has_u32() const {
  return value_case() == kU32;
}
inline bool BasicType::has_u32() const {
  return _internal_has_u32();
}
inline void BasicType::set_has_u32() {
  _oneof_case_[0] = kU32;
}
inline void BasicType::clear_u32() {
  if (_internal_has_u32()) {
    value_.u32_ = 0u;
    clear_has_value();
  }
}
inline uint32_t BasicType::_internal_u32() const {
  if (_internal_has_u32()) {
    return value_.u32_;
  }
  return 0u;
}
inline void BasicType::_internal_set_u32(uint32_t value) {
  if (!_internal_has_u32()) {
    clear_value();
    set_has_u32();
  }
  value_.u32_ = value;
}
inline uint32_t BasicType::u32() const {
  // @@protoc_insertion_point(field_get:tcp_kit.BasicType.u32)
  return _internal_u32();
}
inline void BasicType::set_u32(uint32_t value) {
  _internal_set_u32(value);
  // @@protoc_insertion_point(field_set:tcp_kit.BasicType.u32)
}

// int32 s32 = 2;
inline bool BasicType::_internal_has_s32() const {
  return value_case() == kS32;
}
inline bool BasicType::has_s32() const {
  return _internal_has_s32();
}
inline void BasicType::set_has_s32() {
  _oneof_case_[0] = kS32;
}
inline void BasicType::clear_s32() {
  if (_internal_has_s32()) {
    value_.s32_ = 0;
    clear_has_value();
  }
}
inline int32_t BasicType::_internal_s32() const {
  if (_internal_has_s32()) {
    return value_.s32_;
  }
  return 0;
}
inline void BasicType::_internal_set_s32(int32_t value) {
  if (!_internal_has_s32()) {
    clear_value();
    set_has_s32();
  }
  value_.s32_ = value;
}
inline int32_t BasicType::s32() const {
  // @@protoc_insertion_point(field_get:tcp_kit.BasicType.s32)
  return _internal_s32();
}
inline void BasicType::set_s32(int32_t value) {
  _internal_set_s32(value);
  // @@protoc_insertion_point(field_set:tcp_kit.BasicType.s32)
}

// uint64 u64 = 3;
inline bool BasicType::_internal_has_u64() const {
  return value_case() == kU64;
}
inline bool BasicType::has_u64() const {
  return _internal_has_u64();
}
inline void BasicType::set_has_u64() {
  _oneof_case_[0] = kU64;
}
inline void BasicType::clear_u64() {
  if (_internal_has_u64()) {
    value_.u64_ = uint64_t{0u};
    clear_has_value();
  }
}
inline uint64_t BasicType::_internal_u64() const {
  if (_internal_has_u64()) {
    return value_.u64_;
  }
  return uint64_t{0u};
}
inline void BasicType::_internal_set_u64(uint64_t value) {
  if (!_internal_has_u64()) {
    clear_value();
    set_has_u64();
  }
  value_.u64_ = value;
}
inline uint64_t BasicType::u64() const {
  // @@protoc_insertion_point(field_get:tcp_kit.BasicType.u64)
  return _internal_u64();
}
inline void BasicType::set_u64(uint64_t value) {
  _internal_set_u64(value);
  // @@protoc_insertion_point(field_set:tcp_kit.BasicType.u64)
}

// int64 s64 = 4;
inline bool BasicType::_internal_has_s64() const {
  return value_case() == kS64;
}
inline bool BasicType::has_s64() const {
  return _internal_has_s64();
}
inline void BasicType::set_has_s64() {
  _oneof_case_[0] = kS64;
}
inline void BasicType::clear_s64() {
  if (_internal_has_s64()) {
    value_.s64_ = int64_t{0};
    clear_has_value();
  }
}
inline int64_t BasicType::_internal_s64() const {
  if (_internal_has_s64()) {
    return value_.s64_;
  }
  return int64_t{0};
}
inline void BasicType::_internal_set_s64(int64_t value) {
  if (!_internal_has_s64()) {
    clear_value();
    set_has_s64();
  }
  value_.s64_ = value;
}
inline int64_t BasicType::s64() const {
  // @@protoc_insertion_point(field_get:tcp_kit.BasicType.s64)
  return _internal_s64();
}
inline void BasicType::set_s64(int64_t value) {
  _internal_set_s64(value);
  // @@protoc_insertion_point(field_set:tcp_kit.BasicType.s64)
}

// float f = 5;
inline bool BasicType::_internal_has_f() const {
  return value_case() == kF;
}
inline bool BasicType::has_f() const {
  return _internal_has_f();
}
inline void BasicType::set_has_f() {
  _oneof_case_[0] = kF;
}
inline void BasicType::clear_f() {
  if (_internal_has_f()) {
    value_.f_ = 0;
    clear_has_value();
  }
}
inline float BasicType::_internal_f() const {
  if (_internal_has_f()) {
    return value_.f_;
  }
  return 0;
}
inline void BasicType::_internal_set_f(float value) {
  if (!_internal_has_f()) {
    clear_value();
    set_has_f();
  }
  value_.f_ = value;
}
inline float BasicType::f() const {
  // @@protoc_insertion_point(field_get:tcp_kit.BasicType.f)
  return _internal_f();
}
inline void BasicType::set_f(float value) {
  _internal_set_f(value);
  // @@protoc_insertion_point(field_set:tcp_kit.BasicType.f)
}

// double d = 6;
inline bool BasicType::_internal_has_d() const {
  return value_case() == kD;
}
inline bool BasicType::has_d() const {
  return _internal_has_d();
}
inline void BasicType::set_has_d() {
  _oneof_case_[0] = kD;
}
inline void BasicType::clear_d() {
  if (_internal_has_d()) {
    value_.d_ = 0;
    clear_has_value();
  }
}
inline double BasicType::_internal_d() const {
  if (_internal_has_d()) {
    return value_.d_;
  }
  return 0;
}
inline void BasicType::_internal_set_d(double value) {
  if (!_internal_has_d()) {
    clear_value();
    set_has_d();
  }
  value_.d_ = value;
}
inline double BasicType::d() const {
  // @@protoc_insertion_point(field_get:tcp_kit.BasicType.d)
  return _internal_d();
}
inline void BasicType::set_d(double value) {
  _internal_set_d(value);
  // @@protoc_insertion_point(field_set:tcp_kit.BasicType.d)
}

// bool b = 7;
inline bool BasicType::_internal_has_b() const {
  return value_case() == kB;
}
inline bool BasicType::has_b() const {
  return _internal_has_b();
}
inline void BasicType::set_has_b() {
  _oneof_case_[0] = kB;
}
inline void BasicType::clear_b() {
  if (_internal_has_b()) {
    value_.b_ = false;
    clear_has_value();
  }
}
inline bool BasicType::_internal_b() const {
  if (_internal_has_b()) {
    return value_.b_;
  }
  return false;
}
inline void BasicType::_internal_set_b(bool value) {
  if (!_internal_has_b()) {
    clear_value();
    set_has_b();
  }
  value_.b_ = value;
}
inline bool BasicType::b() const {
  // @@protoc_insertion_point(field_get:tcp_kit.BasicType.b)
  return _internal_b();
}
inline void BasicType::set_b(bool value) {
  _internal_set_b(value);
  // @@protoc_insertion_point(field_set:tcp_kit.BasicType.b)
}

// string str = 8;
inline bool BasicType::_internal_has_str() const {
  return value_case() == kStr;
}
inline bool BasicType::has_str() const {
  return _internal_has_str();
}
inline void BasicType::set_has_str() {
  _oneof_case_[0] = kStr;
}
inline void BasicType::clear_str() {
  if (_internal_has_str()) {
    value_.str_.Destroy();
    clear_has_value();
  }
}
inline const std::string& BasicType::str() const {
  // @@protoc_insertion_point(field_get:tcp_kit.BasicType.str)
  return _internal_str();
}
template <typename ArgT0, typename... ArgT>
inline void BasicType::set_str(ArgT0&& arg0, ArgT... args) {
  if (!_internal_has_str()) {
    clear_value();
    set_has_str();
    value_.str_.InitDefault();
  }
  value_.str_.Set( static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:tcp_kit.BasicType.str)
}
inline std::string* BasicType::mutable_str() {
  std::string* _s = _internal_mutable_str();
  // @@protoc_insertion_point(field_mutable:tcp_kit.BasicType.str)
  return _s;
}
inline const std::string& BasicType::_internal_str() const {
  if (_internal_has_str()) {
    return value_.str_.Get();
  }
  return ::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited();
}
inline void BasicType::_internal_set_str(const std::string& value) {
  if (!_internal_has_str()) {
    clear_value();
    set_has_str();
    value_.str_.InitDefault();
  }
  value_.str_.Set(value, GetArenaForAllocation());
}
inline std::string* BasicType::_internal_mutable_str() {
  if (!_internal_has_str()) {
    clear_value();
    set_has_str();
    value_.str_.InitDefault();
  }
  return value_.str_.Mutable(      GetArenaForAllocation());
}
inline std::string* BasicType::release_str() {
  // @@protoc_insertion_point(field_release:tcp_kit.BasicType.str)
  if (_internal_has_str()) {
    clear_has_value();
    return value_.str_.Release();
  } else {
    return nullptr;
  }
}
inline void BasicType::set_allocated_str(std::string* str) {
  if (has_value()) {
    clear_value();
  }
  if (str != nullptr) {
    set_has_str();
    value_.str_.InitAllocated(str, GetArenaForAllocation());
  }
  // @@protoc_insertion_point(field_set_allocated:tcp_kit.BasicType.str)
}

inline bool BasicType::has_value() const {
  return value_case() != VALUE_NOT_SET;
}
inline void BasicType::clear_has_value() {
  _oneof_case_[0] = VALUE_NOT_SET;
}
inline BasicType::ValueCase BasicType::value_case() const {
  return BasicType::ValueCase(_oneof_case_[0]);
}
// -------------------------------------------------------------------

// GenericMsg

// string api = 1;
inline void GenericMsg::clear_api() {
  api_.ClearToEmpty();
}
inline const std::string& GenericMsg::api() const {
  // @@protoc_insertion_point(field_get:tcp_kit.GenericMsg.api)
  return _internal_api();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void GenericMsg::set_api(ArgT0&& arg0, ArgT... args) {
 
 api_.Set(static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:tcp_kit.GenericMsg.api)
}
inline std::string* GenericMsg::mutable_api() {
  std::string* _s = _internal_mutable_api();
  // @@protoc_insertion_point(field_mutable:tcp_kit.GenericMsg.api)
  return _s;
}
inline const std::string& GenericMsg::_internal_api() const {
  return api_.Get();
}
inline void GenericMsg::_internal_set_api(const std::string& value) {
  
  api_.Set(value, GetArenaForAllocation());
}
inline std::string* GenericMsg::_internal_mutable_api() {
  
  return api_.Mutable(GetArenaForAllocation());
}
inline std::string* GenericMsg::release_api() {
  // @@protoc_insertion_point(field_release:tcp_kit.GenericMsg.api)
  return api_.Release();
}
inline void GenericMsg::set_allocated_api(std::string* api) {
  if (api != nullptr) {
    
  } else {
    
  }
  api_.SetAllocated(api, GetArenaForAllocation());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (api_.IsDefault()) {
    api_.Set("", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  // @@protoc_insertion_point(field_set_allocated:tcp_kit.GenericMsg.api)
}

// repeated .tcp_kit.BasicType params = 2;
inline int GenericMsg::_internal_params_size() const {
  return params_.size();
}
inline int GenericMsg::params_size() const {
  return _internal_params_size();
}
inline void GenericMsg::clear_params() {
  params_.Clear();
}
inline ::tcp_kit::BasicType* GenericMsg::mutable_params(int index) {
  // @@protoc_insertion_point(field_mutable:tcp_kit.GenericMsg.params)
  return params_.Mutable(index);
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::tcp_kit::BasicType >*
GenericMsg::mutable_params() {
  // @@protoc_insertion_point(field_mutable_list:tcp_kit.GenericMsg.params)
  return &params_;
}
inline const ::tcp_kit::BasicType& GenericMsg::_internal_params(int index) const {
  return params_.Get(index);
}
inline const ::tcp_kit::BasicType& GenericMsg::params(int index) const {
  // @@protoc_insertion_point(field_get:tcp_kit.GenericMsg.params)
  return _internal_params(index);
}
inline ::tcp_kit::BasicType* GenericMsg::_internal_add_params() {
  return params_.Add();
}
inline ::tcp_kit::BasicType* GenericMsg::add_params() {
  ::tcp_kit::BasicType* _add = _internal_add_params();
  // @@protoc_insertion_point(field_add:tcp_kit.GenericMsg.params)
  return _add;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::tcp_kit::BasicType >&
GenericMsg::params() const {
  // @@protoc_insertion_point(field_list:tcp_kit.GenericMsg.params)
  return params_;
}

// optional .google.protobuf.Any body = 3;
inline bool GenericMsg::_internal_has_body() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  PROTOBUF_ASSUME(!value || body_ != nullptr);
  return value;
}
inline bool GenericMsg::has_body() const {
  return _internal_has_body();
}
inline const ::PROTOBUF_NAMESPACE_ID::Any& GenericMsg::_internal_body() const {
  const ::PROTOBUF_NAMESPACE_ID::Any* p = body_;
  return p != nullptr ? *p : reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Any&>(
      ::PROTOBUF_NAMESPACE_ID::_Any_default_instance_);
}
inline const ::PROTOBUF_NAMESPACE_ID::Any& GenericMsg::body() const {
  // @@protoc_insertion_point(field_get:tcp_kit.GenericMsg.body)
  return _internal_body();
}
inline void GenericMsg::unsafe_arena_set_allocated_body(
    ::PROTOBUF_NAMESPACE_ID::Any* body) {
  if (GetArenaForAllocation() == nullptr) {
    delete reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(body_);
  }
  body_ = body;
  if (body) {
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:tcp_kit.GenericMsg.body)
}
inline ::PROTOBUF_NAMESPACE_ID::Any* GenericMsg::release_body() {
  _has_bits_[0] &= ~0x00000001u;
  ::PROTOBUF_NAMESPACE_ID::Any* temp = body_;
  body_ = nullptr;
#ifdef PROTOBUF_FORCE_COPY_IN_RELEASE
  auto* old =  reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(temp);
  temp = ::PROTOBUF_NAMESPACE_ID::internal::DuplicateIfNonNull(temp);
  if (GetArenaForAllocation() == nullptr) { delete old; }
#else  // PROTOBUF_FORCE_COPY_IN_RELEASE
  if (GetArenaForAllocation() != nullptr) {
    temp = ::PROTOBUF_NAMESPACE_ID::internal::DuplicateIfNonNull(temp);
  }
#endif  // !PROTOBUF_FORCE_COPY_IN_RELEASE
  return temp;
}
inline ::PROTOBUF_NAMESPACE_ID::Any* GenericMsg::unsafe_arena_release_body() {
  // @@protoc_insertion_point(field_release:tcp_kit.GenericMsg.body)
  _has_bits_[0] &= ~0x00000001u;
  ::PROTOBUF_NAMESPACE_ID::Any* temp = body_;
  body_ = nullptr;
  return temp;
}
inline ::PROTOBUF_NAMESPACE_ID::Any* GenericMsg::_internal_mutable_body() {
  _has_bits_[0] |= 0x00000001u;
  if (body_ == nullptr) {
    auto* p = CreateMaybeMessage<::PROTOBUF_NAMESPACE_ID::Any>(GetArenaForAllocation());
    body_ = p;
  }
  return body_;
}
inline ::PROTOBUF_NAMESPACE_ID::Any* GenericMsg::mutable_body() {
  ::PROTOBUF_NAMESPACE_ID::Any* _msg = _internal_mutable_body();
  // @@protoc_insertion_point(field_mutable:tcp_kit.GenericMsg.body)
  return _msg;
}
inline void GenericMsg::set_allocated_body(::PROTOBUF_NAMESPACE_ID::Any* body) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArenaForAllocation();
  if (message_arena == nullptr) {
    delete reinterpret_cast< ::PROTOBUF_NAMESPACE_ID::MessageLite*>(body_);
  }
  if (body) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena =
        ::PROTOBUF_NAMESPACE_ID::Arena::InternalGetOwningArena(
                reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(body));
    if (message_arena != submessage_arena) {
      body = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, body, submessage_arena);
    }
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  body_ = body;
  // @@protoc_insertion_point(field_set_allocated:tcp_kit.GenericMsg.body)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace tcp_kit

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_generic_5fmsg_2eproto
