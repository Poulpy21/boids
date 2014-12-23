
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
        };
};



#endif /* end of include guard: ARRAYCONTAINER_H */
