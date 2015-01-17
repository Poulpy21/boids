
#ifndef ARRAYCONTAINER_H
#define ARRAYCONTAINER_H

template <typename E>
class ArrayContainer : public AbstractContainer<E> { 
    public:
        ArrayContainer() : AbstractContainer<E>() {}
        ArrayContainer(const ArrayContainer<E> &other) : AbstractContainer<E>(other) {}
        virtual ~ArrayContainer() {}

        void allocate(unsigned int minData) final override {
            this->_data = new E[minData];
            this->_size = minData;
        };
        
        void reallocate(unsigned int minData) final override {
            this->_data = nullptr;
        };
};

template <typename E>
std::ostream & operator << ( std::ostream &os, ArrayContainer<E> array) {
    os << "Array : " << array.elements() << " out of " << array.size() 
        << ", fillrate is " << array.fillRate() << " :" << std::endl;
    for (unsigned int i = 0; i < array.elements(); i++) {
        os << array[i] << ", ";
    }
    os << std::endl;
    return os;
}



#endif /* end of include guard: ARRAYCONTAINER_H */
